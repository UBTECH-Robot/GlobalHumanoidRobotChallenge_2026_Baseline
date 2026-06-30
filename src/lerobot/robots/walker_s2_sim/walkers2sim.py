"""
Walker S2 Isaac Sim 仿真机器人 - LeRobot 0.5.1 实现

使用用例:
    lerobot-record --robot.type=walker_s2_sim --robot.headless=false ...
    lerobot-replay --robot.type=walker_s2_sim --dataset.repo_id=...

功能:
    - 14 自由度双臂控制 (7 关节/臂)
    - 4 相机支持 (head_left, head_right, wrist_left, wrist_right)
    - 键盘遥操作 (通过 WalkerS2KeyboardTeleop)
    - ROS2 遥操作 (可选)
    - Isaac Sim 物理仿真
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from functools import cached_property

import numpy as np
import torch
import yaml

from src.lerobot.robots.robot import Robot
from src.lerobot.robots.config import RobotConfig
from src.lerobot.processor import RobotAction, RobotObservation

from .walkers2simConfig import WalkerS2Config
from .head_stereo_visualizer import HeadStereoVisualizer
from .isaac_sim_robot_interface import IsaacSimRobotInterface, load_config

logger = logging.getLogger(__name__)


@dataclass
class TimingMetric:
    """用于记录性能指标的数据类"""
    count: int = 0
    total_s: float = 0.0
    min_s: float = field(default_factory=lambda: float("inf"))
    max_s: float = 0.0

    def update(self, duration_s: float) -> None:
        duration_s = max(float(duration_s), 0.0)
        self.count += 1
        self.total_s += duration_s
        self.min_s = min(self.min_s, duration_s)
        self.max_s = max(self.max_s, duration_s)

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "avg_s": 0.0, "min_s": 0.0, "max_s": 0.0}
        return {
            "count": self.count,
            "avg_s": self.total_s / self.count,
            "min_s": self.min_s,
            "max_s": self.max_s,
        }





class WalkerS2sim(Robot):
    """
    Walker S2 双臂机器人 Isaac Sim 仿真实现

    功能:
        - 14 自由度双臂控制 (7 关节/臂)
        - 4 相机支持 (head_left, head_right, wrist_left, wrist_right)
        - 支持直接关节位置控制 (回放/推理)
        - 支持末端执行器增量控制 (遥操作 via IK)
        - 夹持器开/关控制
        - 环境物体状态观测 (可选)

    使用例:
        >>> robot = WalkerS2sim(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> action = {"L_shoulder_pitch_joint.pos": 0.1, ...}
        >>> robot.send_action(action)
        >>> robot.disconnect()
    """

    robot_type: str = "walker_s2_sim"
    name: str = "walkerS2"
    config_class: type[RobotConfig] = WalkerS2Config
    CAMERA_NAMES = ["head_left", "head_right", "wrist_left", "wrist_right"]

    # 状态维度：14 臂关节 + 4 手指关节 + 2 夹持器指令 = 20D
    STATE_DIM = 20
    # 动作维度：与状态相同
    ACTION_DIM = 20

    def __init__(self, config: WalkerS2Config | dict | None = None, teleop: Any = None):
        """
        初始化 Walker S2 仿真机器人

        Args:
            config: 机器人配置，可以是 WalkerS2Config 实例、字典或 None
            teleop: 可选的遥操作器实例 (WalkerS2KeyboardTeleop)
        """
        # 处理配置
        if isinstance(config, dict):
            config = WalkerS2Config(**config)
        elif config is None:
            config = WalkerS2Config()

        super().__init__(config)
        self.config = config

        # 加载任务配置
        if self.config.task_cfg_path:
            self.config.task_cfg = load_config(self.config.task_cfg_path)

        # 性能计时
        self._timing_metrics: dict[str, TimingMetric] = {
            "send_action": TimingMetric(),
            "get_observation": TimingMetric(),
            "dt_s": TimingMetric(),
        }

        # Isaac Sim 核心组件
        self._kit: Any = None
        self._world: Any = None
        self._scene_builder: Any = None
        self._robot_interface: Optional[IsaacSimRobotInterface] = None
        self._arm_joint_indices: list[int] = []

        # 遥操作器（外部注入）
        self._teleop: Any = teleop

        # ---- 回调控制相关状态 (移植自 mobile_manipulator.py) ----
        # 回调锁和注册状态
        self._callback_lock = threading.Lock()
        self._callbacks_registered = False

        # Inference 模式：send_action 写入 pending，callback 消费并执行
        self._pending_absolute_action: Optional[np.ndarray] = None
        # 相机图像缓存 (由 render callback 更新)
        self._latest_camera_rgb: dict[str, np.ndarray] = {}
        # 持久化关节目标：统一关节位置控制的核心状态
        self._hold_arm_positions: Optional[np.ndarray] = None
        self._hold_finger_positions: Optional[np.ndarray] = None
        # 夹持模式标记
        self._left_gripping: bool = False
        self._right_gripping: bool = False

        # 步数计数器
        self._send_action_step_idx: int = 0

        # 频率监测（墙钟实测）
        self._phys_cb_count: int = 0
        self._phys_cb_t0: Optional[float] = None
        self._render_cb_count: int = 0
        self._render_cb_t0: Optional[float] = None
        self.measured_physics_hz: float = 0.0
        self.measured_render_hz: float = 0.0

        # 线性插值回到初始位置
        self._go_home = False
        self._num_interpolation_steps = 200
        self._go_home_key_was_pressed = False  # 用于检测按键边缘触发

        # 头部相机可视化
        self._head_visualizer = HeadStereoVisualizer(
            enabled=getattr(self.config, "head_viz_enabled", False),
            window_name=getattr(self.config, "head_viz_window_name", "walker_s2_cameras"),
            scale=getattr(self.config, "head_viz_scale", 1.0),
            every_n=getattr(self.config, "head_viz_every_n", 1),
            window_x=getattr(self.config, "head_viz_window_x", 40),
            window_y=getattr(self.config, "head_viz_window_y", 40),
            show_labels=getattr(self.config, "head_viz_show_labels", True),
        )

        # 环境状态缓存
        self._last_observation: Optional[RobotObservation] = None


    @property
    def cameras(self):
        """返回相机名称列表，兼容框架的 len() 检查"""
        return self.CAMERA_NAMES  # 返回6个相机名称的列表

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接"""
        return self._robot_interface is not None

    @property
    def has_camera(self) -> bool:
        """检查是否有相机"""
        return True

    @property
    def num_cameras(self) -> int:
        """相机数量"""
        return len(self.CAMERA_NAMES)

    @property
    def available_arms(self) -> list[str]:
        """可用的机械臂列表"""
        return ["left", "right"]

    def attach_teleop(self, teleop: Any | None) -> None:
        """绑定外部 teleop，让物理回调可以直接消费键盘状态。"""
        self._teleop = teleop
        if teleop is not None and hasattr(teleop, "enable_callback_mode"):
            teleop.enable_callback_mode()
        elif teleop is None:
            logger.info("WalkerS2sim teleop detached")

    # --- 必须实现的抽象方法 ---
    @property
    def is_calibrated(self) -> bool:
        """仿真机器人无需标定"""
        return True

    def calibrate(self) -> None:
        """
        校准机器人

        Isaac Sim 仿真环境无需校准，此方法为空操作。
        真实机器人实现可能需要重新校准原点或传感器。
        """
        logger.info("Isaac Sim 仿真环境无需校准")
        pass

    def configure(self) -> None:
        """配置机器人 (可选)"""
        pass

    @cached_property
    def _state_features(self) -> dict[str, type]:
        """状态特征定义 (20 维：14 臂关节 + 4 手指关节 + 2 夹持器控制)"""
        return dict.fromkeys(
            (
                "L_shoulder_pitch_joint.pos", "L_shoulder_roll_joint.pos", "L_shoulder_yaw_joint.pos",
                "L_elbow_roll_joint.pos", "L_elbow_yaw_joint.pos", "L_wrist_pitch_joint.pos", "L_wrist_roll_joint.pos",
                "R_shoulder_pitch_joint.pos", "R_shoulder_roll_joint.pos", "R_shoulder_yaw_joint.pos",
                "R_elbow_roll_joint.pos", "R_elbow_yaw_joint.pos", "R_wrist_pitch_joint.pos", "R_wrist_roll_joint.pos",
                "L_finger1_joint.pos", "L_finger2_joint.pos",
                "R_finger1_joint.pos", "R_finger2_joint.pos",
                "left_gripper",  
                "right_gripper",  
            ),
            float,
        )

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        """相机特征定义 - 形状为 (H, W, 3)，与 numpy 图像格式一致"""
        return {
            name: (self.config.camera_height, self.config.camera_width, 3)
            for name in self.CAMERA_NAMES
        }

    @property
    def env_state_dim(self) -> int:
        """环境物体位姿维度 = num_objects * 7 (x, y, z, qx, qy, qz, qw)。"""
        task_cfg = getattr(self.config, "task_cfg", {})
        if not task_cfg:
            return 0

        task_number = task_cfg.get("task_number", 0)
        if task_number == 1:
            part_cfg = task_cfg.get("part", {})
            fallback_count = part_cfg.get("num_parts", 2)
            num_a = max(0, int(part_cfg.get("num_parts_a", fallback_count)))
            num_b = max(0, int(part_cfg.get("num_parts_b", fallback_count)))
            num_objects = num_a + num_b
        elif task_number == 2:
            # Task2 tracks two part types, each with num_parts instances.
            num_objects = task_cfg.get("part", {}).get("num_parts", 5) * 2
        elif task_number == 3:
            num_boxes = len(task_cfg.get('box', {}).get('box_position', []))
            num_parts = task_cfg.get('part', {}).get('num_parts', 3)
            num_objects = num_boxes * num_parts
        elif task_number == 4:
            num_objects = 0
        else:
            num_objects = 0

        return num_objects * 7

    @cached_property
    def _vel_tor_features(self) -> dict[str, type]:
        """关节速度和扭矩特征 (28 维：14 臂关节速度 + 14 臂关节扭矩)"""
        return dict.fromkeys(
            (
                "L_shoulder_pitch_joint.vel", "L_shoulder_roll_joint.vel", "L_shoulder_yaw_joint.vel",
                "L_elbow_roll_joint.vel", "L_elbow_yaw_joint.vel", "L_wrist_pitch_joint.vel", "L_wrist_roll_joint.vel",
                "R_shoulder_pitch_joint.vel", "R_shoulder_roll_joint.vel", "R_shoulder_yaw_joint.vel",
                "R_elbow_roll_joint.vel", "R_elbow_yaw_joint.vel", "R_wrist_pitch_joint.vel", "R_wrist_roll_joint.vel",
                "L_shoulder_pitch_joint.tor", "L_shoulder_roll_joint.tor", "L_shoulder_yaw_joint.tor",
                "L_elbow_roll_joint.tor", "L_elbow_yaw_joint.tor", "L_wrist_pitch_joint.tor", "L_wrist_roll_joint.tor",
                "R_shoulder_pitch_joint.tor", "R_shoulder_roll_joint.tor", "R_shoulder_yaw_joint.tor",
                "R_elbow_roll_joint.tor", "R_elbow_yaw_joint.tor", "R_wrist_pitch_joint.tor", "R_wrist_roll_joint.tor",
            ),
            float,
        )

    @cached_property
    def _env_state_features(self) -> dict[str, type]:
        """环境物体位姿特征定义，每个物体 7 个自由度：x, y, z, qx, qy, qz, qw。

        命名格式：object_1_x, object_1_y, object_1_z, object_1_qx, object_1_qy, object_1_qz, object_1_qw,
                 object_2_x, ...
        """
        env_state_dim = self.env_state_dim
        if env_state_dim == 0:
            return {}

        num_objects = env_state_dim // 7

        # 为每个物体生成 7 个特征名
        features = {}
        for i in range(1, num_objects + 1):
            for suffix in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
                features[f"object_{i}_{suffix}"] = float

        return features

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        观测特征：状态 + 相机 + 环境物体位姿（合并到 state 中）

        Returns:
            包含 20 个机器人状态特征 (float)、4 个相机图像特征 (H, W, 3)
            以及环境物体位姿特征 (object_1_x, object_1_y, ...) 的字典
        """
        return {**self._state_features, **self._camera_features, **self._env_state_features}

    @property
    def action_features(self) -> dict[str, type]:
        """
        动作特征：与状态特征相同 (20 维：14 臂关节 + 4 手指关节 + 2 夹持器控制)

        Returns:
            包含 20 个关节位置特征 (float) 的字典
        """
        return self._state_features

    @property
    def cameras(self):
        """返回相机名称列表，兼容框架的 len() 检查"""
        return self.CAMERA_NAMES

    def record_timing(self, metric_name: str, duration_s: float) -> None:
        """记录性能指标"""
        self._timing_metrics.setdefault(metric_name, TimingMetric()).update(duration_s)

    # ---- 回调注册/注销方法 ----

    def _register_world_callbacks(self) -> None:
        """注册物理和渲染回调到 World"""
        if self._world is None or self._callbacks_registered:
            return

        self._world.add_physics_callback("robot_control", self._robot_control_callback)
        self._world.add_physics_callback("score_input_record", self._score_input_record_callback)
        self._world.add_physics_callback("foam_sync", self._foam_sync_callback)
        self._world.add_render_callback("camera_images", self._camera_images_callback)
        self._callbacks_registered = True
        logger.info("Physics/render callbacks registered")

    def _unregister_world_callbacks(self) -> None:
        """注销所有回调"""
        if self._world is None or not self._callbacks_registered:
            return

        remove_physics = getattr(self._world, "remove_physics_callback", None)
        remove_render = getattr(self._world, "remove_render_callback", None)
        if callable(remove_physics):
            for cb_name in ["robot_control", "score_input_record", "foam_sync"]:
                try:
                    remove_physics(cb_name)
                except Exception:
                    pass
        if callable(remove_render):
            try:
                remove_render("camera_images")
            except Exception:
                pass
        self._callbacks_registered = False

    # ---- 回调实现 ----

    def _robot_control_callback(self, step_size: float) -> None:
        """每个物理步自动执行：统一关节位置控制

        控制逻辑:
        1. 初始化：首次调用时快照当前关节状态作为保持目标
        2. 推理模式：消费 _pending_absolute_action 更新保持目标
        3. 遥操作模式：调用 teleop.get_action_numpy() 获取键盘动作（帧门控在 teleop 内处理）
        4. go_home 模式：检测 toggle_go_home 按键，触发插值回到初始位置
        5. 无输入：持续发出上一帧的保持目标

        注意：帧门控 + 队列合并逻辑在 teleop.get_action_numpy() 内处理，
        因为键盘监听器在 teleop 中，_pressed_keys 和 _keyboard_cmd_queue 由 teleop 管理。
        """
        if not self.is_connected:
            return

        import time as _time
        _now = _time.perf_counter()
        if self._phys_cb_t0 is None:
            self._phys_cb_t0 = _now
        self._phys_cb_count += 1
        _elapsed = _now - self._phys_cb_t0
        if _elapsed >= 1.0:
            self.measured_physics_hz = self._phys_cb_count / _elapsed
            self._phys_cb_count = 0
            self._phys_cb_t0 = _now

        # 检查 go_home 按键（从 teleop 读取按键状态）
        if self._teleop is not None:
            keyboard_state = self._teleop.get_keyboard_state()
            if keyboard_state.get("go_home"):
                # 检测按键按下边缘（防止长按重复触发）
                if not getattr(self, '_go_home_key_was_pressed', False):
                    self._go_home_key_was_pressed = True
                    self._go_home = not self._go_home
                    if self._go_home:
                        logger.info("[go_home] 开始插值回到初始位置...")
                    else:
                        logger.info("[go_home] 取消回到初始位置，恢复正常控制")
            else:
                self._go_home_key_was_pressed = False

        # 初始化保持目标（仅执行一次，快照当前关节状态）
        if self._hold_arm_positions is None:
            states = self._robot_interface.get_joint_states()
            if states:
                self._hold_arm_positions = np.array(states["arm_positions"], dtype=np.float32)
                self._hold_finger_positions = np.array(states["finger_positions"], dtype=np.float32)
                logger.info("[callback] Snapshot initial joint state as hold target")
            else:
                return

        # 读取并消费 Inference mode 的 pending action
        with self._callback_lock:
            abs_action = self._pending_absolute_action
            if abs_action is not None:
                abs_action = abs_action.copy()
                self._pending_absolute_action = None

        if (abs_action is not None) and (not self._go_home):
            # ====== 推理/回放模式：直接使用记录的关节位置 ======
            # action 布局：[0:14]=arm, [14:18]=finger_positions, [18]=left_cmd, [19]=right_cmd
            self._hold_arm_positions = abs_action[:14].copy()
            if abs_action.shape[0] >= 18:
                self._hold_finger_positions = np.array([self._robot_interface.gripper_open_width]*4)
                # self._hold_finger_positions = abs_action[14:18].copy()
            if abs_action.shape[0] >= 20:
                self._left_gripping = float(abs_action[18]) < -0.5
                if self._left_gripping:
                    self._hold_finger_positions[:2] = np.array([self._robot_interface.gripper_close_width]*2)
                self._right_gripping = float(abs_action[19]) < -0.5
                if self._right_gripping:
                    self._hold_finger_positions[2:4] = np.array([self._robot_interface.gripper_close_width]*2)
            print(f"[_robot_control_callback] left_gripping={self._left_gripping}, right_gripping={self._right_gripping}")
        elif not self._go_home:
            # ====== 遥操作模式：通过 teleop 读取键盘状态，计算 IK ======
            if self._teleop is not None:
                # 使用回调模式获取键盘动作
                left_delta, right_delta, left_gripper, right_gripper = self._teleop.get_action_numpy(
                    frame_id=self._send_action_step_idx
                )
                has_left_input = np.linalg.norm(left_delta) > 1e-8
                has_right_input = np.linalg.norm(right_delta) > 1e-8

                if has_left_input or has_right_input:
                    ee_poses = self._robot_interface.get_ee_poses()
                    if ee_poses is not None:
                        left_target = np.asarray(ee_poses["left"][:6]) + left_delta if has_left_input else None
                        right_target = np.asarray(ee_poses["right"][:6]) + right_delta if has_right_input else None

                        ik_result = self._robot_interface.control_dual_arm_ik(
                            step_size=step_size,
                            left_target_xyzrpy=left_target,
                            right_target_xyzrpy=right_target,
                        )
                        if ik_result and "smoothed_positions" in ik_result:
                            sp = ik_result["smoothed_positions"]
                            offset = 0
                            if "left_joint_positions" in ik_result:
                                self._hold_arm_positions[:7] = np.array(sp[offset:offset+7], dtype=np.float32)
                                offset += 7
                            if "right_joint_positions" in ik_result:
                                self._hold_arm_positions[7:14] = np.array(sp[offset:offset+7], dtype=np.float32)

                # 夹持器控制
                gripper_step = 0.002
                g_open = self._robot_interface.gripper_open_width
                g_close = self._robot_interface.gripper_close_width
                g_lo, g_hi = min(g_open, g_close), max(g_open, g_close)
                if abs(left_gripper) > 0.01:
                    self._hold_finger_positions[:2] = np.clip(
                        self._hold_finger_positions[:2] - left_gripper * gripper_step, g_lo, g_hi
                    )
                    self._left_gripping = left_gripper < 0
                if abs(right_gripper) > 0.01:
                    self._hold_finger_positions[2:4] = np.clip(
                        self._hold_finger_positions[2:4] - right_gripper * gripper_step, g_lo, g_hi
                    )
                    self._right_gripping = right_gripper < 0

        else:
            arm_finger_indices = self._robot_interface.arm_joint_indices + self._robot_interface.finger_joint_indices
            if not self._robot_interface.joint_interpolator.interp_active:
                print('[_robot_control_callback] Starting interpolation to initial position...')
                self._robot_interface.joint_interpolator.set_target(
                    start_q=torch.tensor(self._robot_interface.get_joint_states()['all_positions'])[arm_finger_indices],
                    target_q=torch.tensor(self._robot_interface.initial_joint_positions)[arm_finger_indices],
                    num_steps=self._num_interpolation_steps
                )
            arm_finger_positions = self._robot_interface.joint_interpolator.step()  # 执行一步插值
            if isinstance(arm_finger_positions, torch.Tensor):
                arm_finger_positions = arm_finger_positions.detach().cpu().numpy()
            else:
                arm_finger_positions = np.asarray(arm_finger_positions, dtype=np.float32)
            self._hold_arm_positions = arm_finger_positions[:14]
            self._hold_finger_positions = arm_finger_positions[14:18]
            self._left_gripping = False
            self._right_gripping = False
            if self._robot_interface.joint_interpolator.is_finished():
                self._go_home = False  
                all_positions = self._robot_interface.get_joint_states()['all_positions']
                self._robot_interface.reset_ik(all_positions)  
                print('[_robot_control_callback] Interpolation to initial position completed.')      
                      
        # 统一下发保持目标（joint positions 控制）
        self._robot_interface.set_arm_joint_positions(
            target_arm_positions=self._hold_arm_positions.tolist(),
            task_num=self.config.task_cfg.get("task_number", 1)
        )
        self._robot_interface.set_body_joint_positions(
            target_body_positions=0.0,
            task_num=self.config.task_cfg.get("task_number", 1)
        )

        # 夹持器控制：
        #   夹持时：NaN（关闭PD）+ close_tau（纯力矩），避免位置+力矩叠加导致过夹
        #   释放时：open_width（PD开爪）+ open_tau if stuck（主动助力防卡死）
        close_tau = getattr(self._robot_interface, "gripper_close_tau", 200.0)
        open_tau = getattr(self._robot_interface, "gripper_open_tau", -100.0)
        open_width = self._robot_interface.gripper_open_width
        stuck_threshold = 0.005  # 手指实际位置超过 open_width 5mm 以上视为卡住

        # 读取实际手指位置用于卡死检测
        _states = self._robot_interface.get_joint_states()
        actual_finger_pos = (
            np.array(_states["finger_positions"], dtype=np.float32)
            if _states is not None
            else np.array([open_width] * 4, dtype=np.float32)
        )

        gripping = [
            self._left_gripping, self._left_gripping,
            self._right_gripping, self._right_gripping,
        ]
        finger_pos_cmd = []
        efforts = []
        for i, is_gripping in enumerate(gripping):
            if is_gripping:
                finger_pos_cmd.append(float("nan"))  # 关闭PD，不与力矩叠加
                efforts.append(close_tau)
            else:
                finger_pos_cmd.append(open_width)    # PD 驱动开爪
                if actual_finger_pos[i] > open_width + stuck_threshold:
                    efforts.append(open_tau)          # 主动开爪助力（防卡死）
                else:
                    efforts.append(0.0)

        self._robot_interface.set_finger_positions(
            target_fingers=finger_pos_cmd,
            task_num=self.config.task_cfg.get("task_number", 1)
        )
        self._robot_interface.apply_finger_efforts(efforts)

    def _score_input_record_callback(self, step_size: float) -> None:
        """记录分数/目标物体变换"""
        if self._scene_builder is None:
            return
        get_transforms = getattr(self._scene_builder, "get_target_object_transforms", None)
        if callable(get_transforms):
            get_transforms(step_size)

    def _foam_sync_callback(self, _step_size: float) -> None:
        """task4 专用：同步泡沫到箱子"""
        if self._scene_builder is None:
            return
        sync_foam = getattr(self._scene_builder, "sync_foam_to_box", None)
        if callable(sync_foam):
            sync_foam()

    def _camera_images_callback(self, _step: float) -> None:
        """渲染回调：抓取相机图像并缓存"""
        if not self.is_connected:
            return
        import time as _time
        _now = _time.perf_counter()
        if self._render_cb_t0 is None:
            self._render_cb_t0 = _now
        self._render_cb_count += 1
        _elapsed = _now - self._render_cb_t0
        if _elapsed >= 1.0:
            self.measured_render_hz = self._render_cb_count / _elapsed
            self._render_cb_count = 0
            self._render_cb_t0 = _now

        camera_data: dict[str, np.ndarray] = {}
        for cam_name in self.CAMERA_NAMES:
            try:
                rgb = self._robot_interface.get_camera_rgb(cam_name)
                if rgb is not None:
                    camera_data[cam_name] = rgb
            except Exception:
                continue

        if camera_data:
            with self._callback_lock:
                self._latest_camera_rgb = {name: frame.copy() for name, frame in camera_data.items()}
            self._head_visualizer.update_cameras(camera_data)



    def connect(self, calibrate: bool = True) -> None:
        """
        连接机器人并初始化 Isaac Sim 仿真环境

        Args:
            calibrate (bool): 是否自动标定 (仿真环境忽略)

        连接流程:
            1. 创建 SimulationApp
            2. 加载场景 USD
            3. 创建并初始化 World
            4. SceneBuilder 构建场景
            5. 创建机器人接口并初始化
        """
        if self.is_connected:
            logger.info("已经连接")
            return

        if not self.config.task_cfg_path:
            raise ValueError("必须提供 task_cfg_path 以加载场景")

        # 步骤 1: 创建 SimulationApp
        from isaacsim import SimulationApp
        logger.info("步骤 1: 创建 SimulationApp...")
        self._kit = SimulationApp({
            "width": self.config.sim_width,
            "height": self.config.sim_height,
            "headless": self.config.headless,
        })
        logger.info("SimulationApp 创建成功")

        # 步骤 2: 加载场景 USD（关键！之前缺少这一步）
        from isaacsim.core.api import World
        import omni.usd as omni_usd


        logger.info("步骤 2: 加载场景 USD...")
        import os
        scene_path = os.path.join(self.config.task_cfg.get("root_path", ""), self.config.task_cfg.get("scene_usd", ""))
        logger.info(f"场景路径: {scene_path}")
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"场景 USD 文件不存在: {scene_path}")
        
        omni_usd.get_context().open_stage(scene_path)
        logger.info("场景 USD 加载成功")

        # 步骤 3: 创建 World（现在 World 会基于已加载的场景）
        logger.info("步骤 3: 创建 World...")
        if World is None:
            raise ImportError("isaacsim.core.api.World 不可用")
        
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.rendering_dt,
        )
        self._world.initialize_physics()
        logger.info("World 初始化完成")

        # 步骤 4: SceneBuilder 构建场景（添加桌子、零件、箱子等）
        logger.info("步骤 4: SceneBuilder 构建场景...")
        try:
            # 导入 SceneBuilder 和 DataLogger
            # 添加项目根目录到 sys.path，使得 lerobot.Ubtech_sim 可导入
            project_root = Path(__file__).parent.parent.parent.parent.parent
            if str(project_root) not in os.sys.path:
                os.sys.path.append(str(project_root))
                logger.info(f"已将 {project_root} 添加到 sys.path")
            from Ubtech_sim.source.SceneBuilder import SceneBuilder
            from Ubtech_sim.source.DataLogger import DataLogger
            
            # 创建 DataLogger（禁用文件记录）
            data_logger = DataLogger(
                enabled=False,
                csv_path="",
                camera_enabled=False,
                camera_hdf5_path="",
            )
            print("步骤 4: SceneBuilder 构建场景...",self.config.task_cfg)
            self._scene_builder = SceneBuilder(self.config.task_cfg, data_logger=data_logger)
            self._scene_builder.build_all() 
            self._scene_builder.build_robot()
            logger.info("SceneBuilder 场景构建完成")
            
            # 启动仿真
            self._world.play()
            logger.info("World 开始运行")
            # 预热物理引擎，确保 physics_view 创建完成
            for i in range(10):
                self._world.step(render=False)
            logger.info("物理引擎预热完成（10 步）")
        except ImportError as e:
            logger.error(f"无法导入 SceneBuilder: {e}")
            raise
        except Exception as e:
            logger.error(f"场景构建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # 步骤 5: 创建机器人接口（连接到 SceneBuilder 创建的机器人）
        logger.info(f"步骤 5: 创建机器人接口...")

        actual_prim_path = self.config.prim_path

        self._robot_interface = IsaacSimRobotInterface(
            prim_path=actual_prim_path,
            name=self.config.robot_name,
            world=self._world,
            urdf_path=self.config.urdf_path,
        )
        self._robot_interface.initialize()

        # 步骤 5b: 初始化坐标系转换 (after IK solver is ready)
        if self._scene_builder is not None:
            self._scene_builder.init_coordinate_transform(
                self._robot_interface.ik_solver
            )
            logger.info("Coordinate transform initialized")

        # 步骤 6: 快照当前关节状态作为初始保持目标
        states = self._robot_interface.get_joint_states()
        if states:
            self._hold_arm_positions = np.array(states["arm_positions"], dtype=np.float32)
            self._hold_finger_positions = np.array(states["finger_positions"], dtype=np.float32)
            logger.info("快照当前关节状态作为初始保持目标")

        # 步骤 7: 注册回调
        self._register_world_callbacks()
        

        logger.info(f"连接成功！正在控制 {len(self._robot_interface.arm_joint_indices)} 个手臂关节")

    def disconnect(self) -> None:
        """断开连接并清理资源

        清理流程:
            1. 注销回调
            2. 清理机器人接口
            3. 停止 World
            4. 关闭 SimulationApp
        """
        if not self.is_connected:
            return

        # 1. 注销回调
        self._unregister_world_callbacks()

        # 2. 清理机器人接口
        if self._robot_interface:
            self._robot_interface.cleanup()
            self._robot_interface = None

        # 4. 停止 World
        if self._world:
            try:
                self._world.stop()
            except Exception:
                pass
            self._world = None

        # 5. 关闭 SimulationApp
        if self._kit:
            try:
                self._kit.close()
            except Exception:
                pass
            self._kit = None

        logger.info("已断开连接")

    def send_action(self, action: RobotAction | None = None) -> RobotAction:
        """
        发送动作指令到机器人

        Args:
            action (RobotAction | None): 动作字典或 None
                - not None: 推理/回放模式，写入 pending，由 callback 消费执行
                - None: 遥操作模式，callback 直接读取键盘状态完成控制

        Returns:
            RobotAction: 实际执行的动作字典（用于记录），包含 20 个键：
                - 14 臂关节位置
                - 4 手指关节位置
                - 2 夹持器控制指令 (left_gripper, right_gripper)

        控制模式:
            1. 推理/回放模式 (action is not None):
               - 写入 _pending_absolute_action
               - callback 在下一物理步消费并执行
            2. 遥操作模式 (action is None):
               - callback 直接读取键盘状态完成控制
               - 这里仅构建 action 字典用于数据记录
        """
        send_action_start_t = time.perf_counter()
        try:
            if not self.is_connected:
                raise RuntimeError("机器人未连接")

            if action is not None:
                # ====== 模式 A: 推理/回放（统一 20D） ======
                # 写入 pending（callback 将在下一物理步消费）
                # 解析动作字典为 numpy 数组用于验证
                arm_positions = np.array([
                    action[f"L_{j}.pos"] for j in
                    ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_roll_joint", "elbow_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]
                ] + [
                    action[f"R_{j}.pos"] for j in
                    ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_roll_joint", "elbow_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]
                ], dtype=np.float32)

                finger_positions = np.array([
                    action.get("L_finger1_joint.pos", 0.0),
                    action.get("L_finger2_joint.pos", 0.0),
                    action.get("R_finger1_joint.pos", 0.0),
                    action.get("R_finger2_joint.pos", 0.0),
                ], dtype=np.float32)

                left_gripper = action.get("left_gripper", 0.0)
                right_gripper = action.get("right_gripper", 0.0)
                print(f"[send_action] left_gripper={left_gripper}, right_gripper={right_gripper}")
                # 构建 20D action 数组用于验证
                action_np = np.concatenate([
                    arm_positions,
                    finger_positions,
                    np.array([left_gripper, right_gripper], dtype=np.float32)
                ])

                if action_np.shape[0] != self.ACTION_DIM:
                    raise ValueError(f"推理动作 Dimension error: Expected {self.ACTION_DIM}, got {action_np.shape[0]}")

                # 写入 pending（callback 将在下一物理步消费）
                with self._callback_lock:
                    self._pending_absolute_action = action_np.copy()

                # 执行一步物理仿真让 callback 消费 pending action
                self.step(render=True)

                return action

            else:
                # ====== 模式 B: 遥操作（仅构建 action 字典用于记录） ======
                # 实际控制由 callback 通过读取键盘状态完成
                joints_states = self._robot_interface.get_joint_states()
                if joints_states and "arm_positions" in joints_states:
                    arm_pos = joints_states["arm_positions"]
                    finger_pos = joints_states.get("finger_positions", [0.0] * 4)

                    # 构建 action 字典
                    action_dict: RobotAction = {
                        f"L_shoulder_pitch_joint.pos": arm_pos[0],
                        f"L_shoulder_roll_joint.pos": arm_pos[1],
                        f"L_shoulder_yaw_joint.pos": arm_pos[2],
                        f"L_elbow_roll_joint.pos": arm_pos[3],
                        f"L_elbow_yaw_joint.pos": arm_pos[4],
                        f"L_wrist_pitch_joint.pos": arm_pos[5],
                        f"L_wrist_roll_joint.pos": arm_pos[6],
                        f"R_shoulder_pitch_joint.pos": arm_pos[7],
                        f"R_shoulder_roll_joint.pos": arm_pos[8],
                        f"R_shoulder_yaw_joint.pos": arm_pos[9],
                        f"R_elbow_roll_joint.pos": arm_pos[10],
                        f"R_elbow_yaw_joint.pos": arm_pos[11],
                        f"R_wrist_pitch_joint.pos": arm_pos[12],
                        f"R_wrist_roll_joint.pos": arm_pos[13],
                        "L_finger1_joint.pos": finger_pos[0],
                        "L_finger2_joint.pos": finger_pos[1],
                        "R_finger1_joint.pos": finger_pos[2],
                        "R_finger2_joint.pos": finger_pos[3],
                        "left_gripper": -1.0 if self._left_gripping else 1.0,
                        "right_gripper": -1.0 if self._right_gripping else 1.0,
                    }
                else:
                    raise RuntimeError("无法获取关节状态以构建 action 字典")
                # 执行一步物理仿真
                self.step(render=True)

                return action_dict

        finally:
            duration_s = time.perf_counter() - send_action_start_t
            self.record_timing("send_action", duration_s)

    def step(self, render: bool = True) -> None:
        """推进仿真一步

        Args:
            render: 是否渲染图像，默认 True
        """
        if self._world:
            self._world.step(render=render)
            self._send_action_step_idx += 1

    def get_observation(self) -> RobotObservation:
        """
        获取机器人观测 (关节状态 + 相机 RGB + 环境状态)

        Returns:
            RobotObservation: 扁平字典，key 与 observation_features 完全匹配:
                - 14 臂关节位置 (float): L_shoulder_pitch_joint.pos, ..., R_wrist_roll_joint.pos
                - 4 手指关节位置 (float): L_finger1_joint.pos, L_finger2_joint.pos, R_finger1_joint.pos, R_finger2_joint.pos
                - 2 夹持器控制 (float): left_gripper, right_gripper
                - 4 相机 RGB 图像 (H, W, 3): head_left, head_right, wrist_left, wrist_right
                - 可选环境状态向量 (N,): observation.environment_state
        """
        start_t = time.perf_counter()
        try:
            if not self.is_connected:
                raise RuntimeError("机器人未连接")

            obs: RobotObservation = {}

            # 获取关节状态 - 扁平 key 与 observation_features 匹配
            joints_states = self._robot_interface.get_joint_states()
            if joints_states and 'arm_positions' in joints_states:
                arm_pos = joints_states['arm_positions']
                finger_pos = joints_states.get('finger_positions', [0.0] * 4)

                # 14 臂关节
                arm_joint_names = [
                    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
                    "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
                    "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
                    "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
                ]
                for i, joint_name in enumerate(arm_joint_names):
                    obs[f"{joint_name}.pos"] = torch.tensor(arm_pos[i], dtype=torch.float32)

                # 14 臂关节速度和扭矩（侧录用，不进 observation.state）
                arm_vel = joints_states.get('arm_velocities', [0.0] * 14)
                arm_tau = joints_states.get('arm_torques', [0.0] * 14)
                for i, joint_name in enumerate(arm_joint_names):
                    obs[f"_vel_{joint_name}"] = torch.tensor(arm_vel[i], dtype=torch.float32)
                    obs[f"_tor_{joint_name}"] = torch.tensor(arm_tau[i], dtype=torch.float32)

                # 4 手指关节
                finger_joint_names = ["L_finger1_joint", "L_finger2_joint", "R_finger1_joint", "R_finger2_joint"]
                for i, joint_name in enumerate(finger_joint_names):
                    obs[f"{joint_name}.pos"] = torch.tensor(finger_pos[i], dtype=torch.float32)

                # 2 夹持器控制
                obs["left_gripper"] = torch.tensor(-1.0 if self._left_gripping else 1.0, dtype=torch.float32)
                obs["right_gripper"] = torch.tensor(-1.0 if self._right_gripping else 1.0, dtype=torch.float32)
            else:
                raise RuntimeError("无法获取关节状态")

            # 获取相机图像 - key 与 observation_features 匹配，形状为 (H, W, 3)
            for cam_name in self.CAMERA_NAMES:
                try:
                    rgbd = self._robot_interface.get_camera_rgbd(cam_name)
                    if rgbd and rgbd.get("rgb") is not None:
                        # 保持 (H, W, 3) 格式，不做 permute
                        img = torch.from_numpy(rgbd["rgb"]).float() / 255.0
                        obs[cam_name] = img
                    else:
                        logger.warning(f"相机 {cam_name} 无法获取 RGB 图像")
                        h, w = self.config.camera_height, self.config.camera_width
                        obs[cam_name] = torch.zeros(h, w, 3, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"获取相机 {cam_name} 图像失败：{e}")
                    h, w = self.config.camera_height, self.config.camera_width
                    obs[cam_name] = torch.zeros(h, w, 3, dtype=torch.float32)

            # 获取环境物体位姿 - 作为独立的 object_1_x, object_1_y, ... 键添加到 obs 中
            # 这些特征会被合并到 observation.state 中
            env_state_dim = self.env_state_dim
            if env_state_dim > 0:
                try:
                    if self._scene_builder is None:
                        raise RuntimeError("SceneBuilder 未初始化")

                    env_state_np = np.asarray(self._scene_builder.get_object_poses_flat(), dtype=np.float32).reshape(-1)
                    if env_state_np.shape[0] != env_state_dim:
                        raise RuntimeError(
                            f"环境状态维度不匹配：期望 {env_state_dim}, 实际 {env_state_np.shape[0]}"
                        )

                    # 将扁平向量分解为独立的 object_i_x, object_i_y, ... 键
                    # 格式：[obj1_x, obj1_y, obj1_z, obj1_qx, obj1_qy, obj1_qz, obj1_qw, obj2_x, ...]
                    num_objects = env_state_dim // 7
                    for i in range(1, num_objects + 1):
                        base_idx = (i - 1) * 7
                        obs[f"object_{i}_x"] = torch.tensor(env_state_np[base_idx], dtype=torch.float32)
                        obs[f"object_{i}_y"] = torch.tensor(env_state_np[base_idx + 1], dtype=torch.float32)
                        obs[f"object_{i}_z"] = torch.tensor(env_state_np[base_idx + 2], dtype=torch.float32)
                        obs[f"object_{i}_qx"] = torch.tensor(env_state_np[base_idx + 3], dtype=torch.float32)
                        obs[f"object_{i}_qy"] = torch.tensor(env_state_np[base_idx + 4], dtype=torch.float32)
                        obs[f"object_{i}_qz"] = torch.tensor(env_state_np[base_idx + 5], dtype=torch.float32)
                        obs[f"object_{i}_qw"] = torch.tensor(env_state_np[base_idx + 6], dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"获取环境物体位姿失败：{e}")
                    # 创建零值特征
                    num_objects = env_state_dim // 7
                    for i in range(1, num_objects + 1):
                        obs[f"object_{i}_x"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_y"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_z"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qx"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qy"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qz"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qw"] = torch.tensor(0.0, dtype=torch.float32)

            self._last_observation = obs
            return obs

        finally:
            self._timing_metrics["get_observation"].update(time.perf_counter() - start_t)



    def log_ee_poses(self) -> None:
        """打印双臂末端姿态 (xyzrpy)，用于遥操作实时监控"""
        if not self.is_connected or self._robot_interface is None:
            return
        try:
            ee_poses = self._robot_interface.get_ee_poses()
            if ee_poses is None:
                return
            left = ee_poses.get("left")
            right = ee_poses.get("right")
            if left is not None and right is not None:
                print(
                    f"[EE] L: xyz=({left[0]:.4f}, {left[1]:.4f}, {left[2]:.4f}) "
                    f"rpy=({left[3]:.4f}, {left[4]:.4f}, {left[5]:.4f}) | "
                    f"R: xyz=({right[0]:.4f}, {right[1]:.4f}, {right[2]:.4f}) "
                    f"rpy=({right[3]:.4f}, {right[4]:.4f}, {right[5]:.4f})"
                )
        except Exception as e:
            logger.warning(f"log_ee_poses failed: {e}")

    def print_logs(self) -> None:
        """打印机器人当前状态信息"""
        if not self.is_connected:
            print("未连接")
            return

        print(f"当前控制臂：{getattr(self, 'current_control_arm', 'N/A')}")
        print(f"双臂同步模式：{getattr(self, 'bimanual_control_enabled', False)}")
        if self._hold_arm_positions is not None:
            print(f"保持位置：{self._hold_arm_positions.tolist()[:6]}...")  # 只显示前 6 个

    def reset(self) -> None:
        """重置环境：场景物体恢复初始 Pose/随机化，机器人恢复初始关节，控制接口保持不变。

        重置流程:
            1. 取消注册回调（防止 reset 过程中 callback 干扰）
            2. 重置场景（SceneBuilder.reset()）
            3. 任务 1/3 需要 world.reset() 重建物理视图
            4. 重置机器人关节到初始 Pose
            5. 推进物理仿真让新 Pose 生效 + 物理稳定
            6. 重置步数计数器
            7. 清空 pending 控制状态和键盘信号队列
            8. 重新快照 joint states 作为保持目标
            9. 重新注册回调
        """
        if self._scene_builder is None:
            logger.warning("SceneBuilder 未初始化，无法重置")
            return

        # 1. 取消注册回调
        self._unregister_world_callbacks()

        # 2. 重置场景
        self._scene_builder.reset()

        # 3. 任务 1/3 删除了旧 prim 并创建新 prim，物理视图失效，需要 world.reset() 重建
        task_num = self.config.task_cfg.get("task_number", 0)
        if task_num in (1, 3):
            logger.info("[reset] 重新初始化物理仿真 (due to prim deletion/creation)...")
            if self._world is not None:
                self._world.reset()
                scatter_after_reset = getattr(self._scene_builder, "scatter_after_reset", None)
                if callable(scatter_after_reset):
                    scatter_after_reset()

        # 4. 重置机器人关节到初始 Pose
        if self._robot_interface is not None:
            self._robot_interface.reset()

        # 5. 推进物理仿真，让新 Pose 生效 + 物理稳定
        settle_steps = 5
        for _ in range(settle_steps):
            if self._robot_interface is not None and self._robot_interface._world is not None:
                self._robot_interface._world.step(render=True)

        # 6. 重置步数计数器
        self._send_action_step_idx = 0

        # 7. 清空 pending 控制状态
        with self._callback_lock:
            self._pending_absolute_action = None
            self._latest_camera_rgb = {}

        # 重置 teleop 键盘状态
        if self._teleop is not None:
            self._teleop.reset()

        self._left_gripping = False
        self._right_gripping = False
        self._go_home = False  # 重置回家标志
        self._go_home_key_was_pressed = False  # 重置回家按键状态

        # 8. 重新快照 joint states 作为保持目标
        states = self._robot_interface.get_joint_states()
        if states:
            self._hold_arm_positions = np.array(self._robot_interface.arm_joint_initial_positions, dtype=np.float32)
            self._hold_finger_positions = np.array(self._robot_interface.finger_joint_initial_positions, dtype=np.float32)
        else:
            self._hold_arm_positions = None
            self._hold_finger_positions = None

        # 9. 重新注册回调
        self._register_world_callbacks()

        logger.info("[WalkerS2sim] Environment reset complete")

    def set_environment_state(self, env_state: np.ndarray | torch.Tensor | list[float]) -> None:
        """按扁平环境状态向量恢复仿真物体位姿。"""
        expected_dim = self.env_state_dim
        env_state_np = np.asarray(env_state, dtype=np.float32).reshape(-1)
        if env_state_np.shape[0] != expected_dim:
            raise ValueError(f"环境状态维度不匹配: 期望 {expected_dim}, 实际 {env_state_np.shape[0]}")

        callbacks_registered = self._callbacks_registered
        if callbacks_registered:
            self._unregister_world_callbacks()

        try:
            self._scene_builder.set_object_poses_from_flat(env_state_np)
            if self._world is not None:
                settle_steps = max(1, int(0.2 / self.config.physics_dt))
                for _ in range(settle_steps):
                    self._world.step(render=False)
        finally:
            if callbacks_registered:
                self._register_world_callbacks()

        self._last_observation = None
        logger.info("[WalkerS2sim] Environment state restored from SceneBuilder")
