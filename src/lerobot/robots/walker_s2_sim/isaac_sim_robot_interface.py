"""Isaac Sim robot interface: Direct connection to Articulation created by SceneBuilder.

Contains:
- load_config — Load task YAML and resolve root_path to absolute path
- IsaacSimRobotInterface — Articulation wrapper, IK, camera, gripper control
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# =========================================================================
# YAML Task configuration loading
# =========================================================================


def load_config(config_path: str) -> dict:
    """
    Load YAML task config file and resolve relative paths.

    Args:
        config_path: Absolute or relative path to YAML config file.

    Returns:
        Parsed config dict with ``root_path`` resolved to absolute path.
    """
    config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(config_path)

    if "root_path" in cfg:
        cfg["root_path"] = os.path.abspath(
            os.path.join(yaml_dir, cfg["root_path"])
        )
    return cfg


# =========================================================================
# Joint interpolator interface: Linear interpolation for joint position commands
# =========================================================================
class JointInterpolator:
    """关节角线性插值器（LERP），适用于 Isaac Lab 仿真逐帧执行"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置插值状态"""
        self.current_step = 0
        self.total_steps = 0
        self.start_q = None
        self.target_q = None
        self.interp_active = False

    def set_target(self, start_q: torch.Tensor, target_q: torch.Tensor, num_steps: int = 10):
        """
        设置插值起点、终点、总步数
        Args:
            start_q: 初始关节角 (N, num_dofs)
            target_q: 目标关节角 (N, num_dofs)
            num_steps: 插值总步数（多少步到达目标）
        """
        if num_steps <= 0:
            self.interp_active = False
            return

        self.start_q = start_q.clone()
        self.target_q = target_q.clone()
        self.total_steps = num_steps
        self.current_step = 0
        self.interp_active = True

    def step(self) -> torch.Tensor:
        """
        执行一帧插值，返回当前应该到达的关节角
        Returns:
            插值后的关节角 (N, num_dofs)
        """
        # 安全判断：未设置目标则抛出异常
        if self.start_q is None or self.target_q is None:
            raise RuntimeError("JointInterpolator: 请先调用 set_target() 设置起始和目标关节角")

        # 如果没有插值任务，直接返回目标
        if not self.interp_active:
            return self.target_q.clone()

        # 当前步归一化比例 [0, 1]
        alpha = float(self.current_step) / max(self.total_steps - 1, 1)
        alpha = float(max(0.0, min(1.0, alpha)))

        # 线性插值核心公式
        q_interp = self.start_q + alpha * (self.target_q - self.start_q)

        # 步进计数
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.interp_active = False

        return q_interp

    def is_finished(self) -> bool:
        """判断插值是否完成"""
        return not self.interp_active






# =========================================================================
# 笛卡尔空间轨迹规划：位置LERP + 姿态Slerp
# =========================================================================
class CartesianTrajectoryPlanner:
    """
    笛卡尔空间轨迹规划
    - 位置：线性插值 LERP
    - 姿态：RPY → Rotation → scipy Slerp → RPY
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_steps = 0
        self.start_pose = None  # [x,y,z, r,p,y]
        self.target_pose = None
        self.slerp = None        # scipy Slerp 对象
        self.traj_active = False

    def set_target(
        self,
        start_pose: np.ndarray,
        target_pose: np.ndarray,
        num_steps: int = 50
    ):
        if num_steps <= 0:
            self.traj_active = False
            return

        self.start_pose = np.array(start_pose, dtype=np.float32)
        self.target_pose = np.array(target_pose, dtype=np.float32)
        self.total_steps = num_steps
        self.current_step = 0

        # RPY → Rotation → Slerp
        rpy0 = self.start_pose[3:]
        rpy1 = self.target_pose[3:]
        rot0 = Rotation.from_euler("xyz", rpy0)
        rot1 = Rotation.from_euler("xyz", rpy1)

        # 构造scipy Slerp（关键帧时间0和1）
        self.slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot0, rot1]))
        self.traj_active = True

    def step(self) -> np.ndarray:
        if not self.traj_active or self.start_pose is None or self.target_pose is None:
            raise RuntimeError("先调用 set_target() 设置起始和目标姿态")

        alpha = float(self.current_step) / max(self.total_steps - 1, 1)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 位置 LERP
        pos_interp = self.start_pose[:3] + alpha * (self.target_pose[:3] - self.start_pose[:3])

        # 姿态 Slerp（直接用scipy）
        rot_interp = self.slerp(alpha)
        rpy_interp = rot_interp.as_euler("xyz")

        current_pose = np.concatenate([pos_interp, rpy_interp])

        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.traj_active = False

        return current_pose

    def is_finished(self) -> bool:
        return not self.traj_active


# =========================================================================
# Robot interface (simplified version, directly uses robot created by SceneBuilder)
# =========================================================================


class IsaacSimRobotInterface:
    """
    Robot interface, directly connects to robot created by SceneBuilder
    """

    arm_joint_names: list[str] = [
        "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
        "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
        "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
        "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
    ]

    finger_joint_names: list[str] = [
        "L_finger1_joint",
        "L_finger2_joint",
        "R_finger1_joint",
        "R_finger2_joint",
    ]

    head_joint_names: list[str] = [
        "head_pitch_joint",
        "head_yaw_joint",
    ]

    gripper_open_width = -0.0215
    gripper_close_width = 0.01
    gripper_open_tau = -100.0
    gripper_close_tau = 200.0

    sixforce_joint_names: list[str] = [
        "L_sixforce_joint",
        "R_sixforce_joint",
    ]

    def __init__(
        self,
        prim_path: str,
        name: str = "walkerS2",
        world: Any = None,
        urdf_path: Optional[str] = None,
    ):
        self.prim_path = prim_path
        self.name = name
        self._world = world
        self._articulation = None
        self.time = 0.0
        self.urdf_path = urdf_path

        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.body_joint_indices: list[int] = []
        
        self.sixforce_joint_names = ["L_sixforce_joint", "R_sixforce_joint"]
        self.cameras = {}

        self._camera_prim_paths = {
            'head_left': f"{prim_path}/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01",
            'head_right': f"{prim_path}/head_pitch_link/head_stereo_right/head_stereo_right_Camera_01",
            'wrist_left': f"{prim_path}/L_camera_link/L_camera_link/L_wrist_camera/L_wrist_Camera",
            'wrist_right': f"{prim_path}/R_camera_link/R_camera_link/R_wrist_camera/R_wrist_Camera",
        }

        self.initial_joint_positions = None
        self.ik_solver: Optional[Any] = None
        self._left_arm_isaac_indices = None
        self._right_arm_isaac_indices = None
        self._waist_isaac_indices = None
        self._waist_init_positions = None
        self._ik_warn_counter = 0
        self._smooth_alpha = 0.3
        self._last_arm_positions = {}
        self._joint_value_map = {
            "L_elbow_roll_joint": -1.8963565338596158,
            "L_elbow_yaw_joint": 1.4000461262831179,
            "L_shoulder_pitch_joint": 0.09322471888572098,
            "L_shoulder_roll_joint": -0.5933223843430208,
            "L_shoulder_yaw_joint": -1.595878574835185,
            "L_wrist_pitch_joint": -0.00048740902645395785,
            "L_wrist_roll_joint": 0.0998718010009366,
            "R_elbow_roll_joint": -1.8963607249359917,
            "R_elbow_yaw_joint": -1.4000874256427638,
            "R_shoulder_pitch_joint": -0.09321727661087699,
            "R_shoulder_roll_joint": -0.5933455607833843,
            "R_shoulder_yaw_joint": 1.595869459316937,
            "R_wrist_pitch_joint": 0.00048144049606466176,
            "R_wrist_roll_joint": 0.09985407619802703,
            "head_pitch_joint": -0.785398163,
            "head_yaw_joint": 1.9677590016147396e-07,
        }

        # 关节插值器
        self.joint_interpolator = JointInterpolator()
        
        # 笛卡尔空间轨迹规划器
        self.cartesian_planner = CartesianTrajectoryPlanner()

    def initialize(self):
        """Initialize articulation"""
        from isaacsim.core.prims import Articulation
        if Articulation is None:
            raise ImportError("isaacsim.core.prims.Articulation 不可用")

        logger.info(f"Connecting to Articulation: {self.prim_path}")

        self._articulation = Articulation(
            prim_paths_expr=self.prim_path,
            name=self.name,
        )
        self._articulation.initialize()

        all_joint_names = self._articulation.dof_names
        logger.info(f"Total robot joints: {len(all_joint_names)}")
        logger.info(f"All joints: {all_joint_names}")

        arm_missing_joints = []
        finger_missing_joints = []

        for arm_joint in self.arm_joint_names:
            if arm_joint in all_joint_names:
                idx = all_joint_names.index(arm_joint)
                self.arm_joint_indices.append(idx)
                logger.info(f"  [{len(self.arm_joint_indices)-1}] {arm_joint} -> global index {idx}")
            else:
                arm_missing_joints.append(arm_joint)
                logger.error(f"  Joint not found: {arm_joint}")

        if arm_missing_joints:
            raise RuntimeError(f"Missing {len(arm_missing_joints)} requiredarm joints: {arm_missing_joints}")

        if len(self.arm_joint_indices) != 14:
            raise RuntimeError(f"arm jointscount error: Expected14, got {len(self.arm_joint_indices)} ")

        for finger_joint in self.finger_joint_names:
            if finger_joint in all_joint_names:
                idx = all_joint_names.index(finger_joint)
                self.finger_joint_indices.append(idx)
                logger.info(f"  [{len(self.finger_joint_indices)-1}] {finger_joint} -> global index {idx}")
            else:
                finger_missing_joints.append(finger_joint)
                logger.error(f"  Joint not found: {finger_joint}")
                logger.warning(f"未找到Finger joint: {finger_joint}，Gripper控制将不可用")
        if finger_missing_joints:
            logger.warning(f"Missing {len(finger_missing_joints)} Finger joint: {finger_missing_joints}")

        if len(self.finger_joint_indices) != 4:
            raise RuntimeError(f"Finger jointcount error: Expected4, got {len(self.finger_joint_indices)} ")

        head_joint_indices = [
            all_joint_names.index(joint_name)
            for joint_name in self.head_joint_names
            if joint_name in all_joint_names
        ]
        used_indices = set(self.arm_joint_indices + self.finger_joint_indices + head_joint_indices)
        self.body_joint_indices = [idx for idx in range(len(all_joint_names)) if idx not in used_indices]
        logger.info(f"Body joints (exclude arm+finger+head): {len(self.body_joint_indices)}")

        self.initial_joint_positions = [0.0] * len(all_joint_names)

        arm_initial_values = [
            0.09322471888572098,      # L_shoulder_pitch_joint
            -0.5933223843430208,      # L_shoulder_roll_joint
            -1.595878574835185,       # L_shoulder_yaw_joint
            -1.8963565338596158,      # L_elbow_roll_joint
            1.4000461262831179,       # L_elbow_yaw_joint
            -0.00048740902645395785,  # L_wrist_pitch_joint
            0.0998718010009366,       # L_wrist_roll_joint
            -0.09321727661087699,     # R_shoulder_pitch_joint
            -0.5933455607833843,      # R_shoulder_roll_joint
            1.595869459316937,        # R_shoulder_yaw_joint
            -1.8963607249359917,      # R_elbow_roll_joint
            -1.4000874256427638,      # R_elbow_yaw_joint
            0.00048144049606466176,   # R_wrist_pitch_joint
            0.09985407619802703,      # R_wrist_roll_joint
        ]
        self.finger_joint_initial_positions = [self.gripper_open_width] * 4
        self.arm_joint_initial_positions = arm_initial_values
        for i, global_idx in enumerate(self.arm_joint_indices):
            self.initial_joint_positions[global_idx] = arm_initial_values[i]

        extra_defaults = {
            "head_pitch_joint": -0.785398163,
            "head_yaw_joint": 1.9677590016147396e-07,
        }
        for joint_name, value in extra_defaults.items():
            if joint_name in all_joint_names:
                idx = all_joint_names.index(joint_name)
                self.initial_joint_positions[idx] = value
        all_indices = list(range(len(all_joint_names)))
        self._all_dof_indices = all_indices
        self._articulation.set_joint_positions(
            torch.tensor(self.initial_joint_positions),
            joint_indices=torch.tensor(all_indices)
        )

        if self._world is not None:
            self._world.step(render=False)
            logger.info("Executing initial simulation step")

        self._setup_cameras()

        self.initialize_ik(urdf_path=self.urdf_path)
        logger.info(f"Robot initialization complete, controlling {len(self.arm_joint_indices)} arm joints")

    def _setup_cameras(self):
        """Setup cameras"""
        from isaacsim.sensors.camera import Camera
        if Camera is None:
            logger.warning("isaacsim.sensors.camera.Camera unavailable, camera function disabled")
            return

        for cam_name, prim_path in self._camera_prim_paths.items():
            try:
                self.cameras[cam_name] = Camera(prim_path=prim_path, resolution=(640, 480))
                self.cameras[cam_name].initialize()
                self.cameras[cam_name].add_distance_to_image_plane_to_frame()
                logger.info(f"Camera {cam_name} initialized successfully")
            except Exception as e:
                logger.warning(f"Camera {cam_name} 初始化failed: {e}")

        self.cameras['dummy_camera_top'] = Camera(
            prim_path="/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/dummy_camera_top",
            translation=[-2.54145, -0.06363, 2.4821],
            orientation=[0.942732, -0.008441, 0.333388, 0.006151]
        )
        self.cameras['dummy_camera_top'].initialize()
        self.cameras['dummy_camera_top'].add_distance_to_image_plane_to_frame()

        self.cameras['dummy_camera_side'] = Camera(
            prim_path="/Replicator/Ref_Xform/Ref/dummy_camera_side",
            translation=[2.06555, -0.02631, 0.95453],
            orientation=[-5.94300e-03, -3.24760e-02, -2.01000e-04, 9.99455e-01]
        )
        self.cameras['dummy_camera_side'].initialize()
        self.cameras['dummy_camera_side'].add_distance_to_image_plane_to_frame()

    def reinitialize_articulation(self):
        """Rebuild Articulation object after scene reset to restore _physics_view.

        In Isaac Sim, once USD stage changes (e.g., RemovePrim / rep.create),
        The old Articulation object's _physics_view is destroyed and cannot be restored
        by calling initialize() again. Must create a brand new Articulation wrapper.
        """
        from isaacsim.core.prims import Articulation
        logger.info("[reinitialize_articulation] Rebuilding Articulation ...")
        self._articulation = Articulation(
            prim_paths_expr=self.prim_path,
            name=self.name,
        )
        self._articulation.initialize()
        logger.info("[reinitialize_articulation] Articulation rebuilt complete")


    def reset_ik(self,current_joint_positions: Optional[list[float]] = None):
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            self.ik_solver.sync_joint_positions(
                self._articulation.dof_names,
                current_joint_positions,
            )
            self.ik_solver.save_initial_q()

    def reset(self):
        """将机器人重置到初始化时的关节初始Pose，并同步重置IK相关状态。

        Reset includes:
        1. All joint positions return to initial values set by ``initialize()``
        2. Zero out all joint velocities when possible
        3. 清空IK平滑缓存、告警计数和累计时间
        4. 若IK已初始化，则重建其warm-start初始参考
        """
        initial_positions = np.asarray(self.initial_joint_positions, dtype=np.float32)
        all_indices = np.asarray(self._all_dof_indices, dtype=np.int32)
        self._articulation.set_joint_positions(
            torch.tensor(initial_positions, dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(all_indices), dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self.time = 0.0
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            self.ik_solver.sync_joint_positions(
                self._articulation.dof_names,
                initial_positions.tolist(),
            )
            self.ik_solver.save_initial_q()

        print("[RobotArticulation] Robot reset to initial pose complete")

    def cleanup(self):
        """Cleanup resources"""
        if self._articulation is not None:
            try:
                if self.initial_joint_positions is not None:
                    all_joint_names = self._articulation.dof_names
                    all_indices = list(range(len(all_joint_names)))
                    self._articulation.set_joint_positions(
                        torch.tensor(self.initial_joint_positions),
                        joint_indices=torch.tensor(all_indices)
                    )
            except Exception:
                pass
            self._articulation = None
        self.cameras.clear()

    def get_joint_states(self):
        """获取joint states（Return所有34关节，但只提取14arm joints）"""
        if self._articulation is None:
            return None

        try:
            all_names = self._articulation.dof_names
            all_joint_positions = self._articulation.get_joint_positions()
            all_joint_velocities = self._articulation.get_joint_velocities()
            all_joint_efforts = self._articulation.get_measured_joint_efforts()

            if hasattr(all_joint_positions, 'shape') and len(all_joint_positions.shape) > 1:
                all_joint_positions = all_joint_positions.flatten()
            if hasattr(all_joint_velocities, 'shape') and len(all_joint_velocities.shape) > 1:
                all_joint_velocities = all_joint_velocities.flatten()
            if hasattr(all_joint_efforts, 'shape') and len(all_joint_efforts.shape) > 1:
                all_joint_efforts = all_joint_efforts.flatten()

            if isinstance(all_joint_positions, np.ndarray):
                all_joint_positions = all_joint_positions.flatten()
            if isinstance(all_joint_velocities, np.ndarray):
                all_joint_velocities = all_joint_velocities.flatten()
            if isinstance(all_joint_efforts, np.ndarray):
                all_joint_efforts = all_joint_efforts.flatten()

            if not isinstance(all_joint_positions, torch.Tensor):
                all_joint_positions = torch.tensor(all_joint_positions, dtype=torch.float32)
            if not isinstance(all_joint_velocities, torch.Tensor):
                all_joint_velocities = torch.tensor(all_joint_velocities, dtype=torch.float32)
            if not isinstance(all_joint_efforts, torch.Tensor):
                all_joint_efforts = torch.tensor(all_joint_efforts, dtype=torch.float32)

            all_joint_positions = all_joint_positions.flatten()
            all_joint_velocities = all_joint_velocities.flatten()
            all_joint_efforts = all_joint_efforts.flatten()

            if len(all_joint_positions) != len(all_names):
                return None

            arm_indices = torch.tensor(self.arm_joint_indices, dtype=torch.long)
            arm_positions = all_joint_positions[arm_indices]
            arm_vel = all_joint_velocities[arm_indices]
            arm_tau = all_joint_efforts[arm_indices]

            finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.long)
            finger_positions = all_joint_positions[finger_indices]
            finger_tau = all_joint_efforts[finger_indices]
            finger_vel = all_joint_velocities[finger_indices]

            return {
                'all_names': all_names,
                'all_positions': all_joint_positions.tolist(),
                'arm_names': self.arm_joint_names,
                'arm_positions': arm_positions.tolist(),
                'arm_indices': self.arm_joint_indices,
                'arm_velocities': arm_vel.tolist(),
                'arm_torques': arm_tau.tolist(),
                'finger_names': self.finger_joint_names,
                'finger_positions': finger_positions.tolist(),
                'finger_indices': self.finger_joint_indices,
                'finger_velocities': finger_vel.tolist(),
                'finger_torques': finger_tau.tolist(),
            }
        except Exception as e:
            logger.error(f"[get_joint_states] Exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def set_arm_joint_positions(self, target_arm_positions, task_num: int = None):
        """设置14arm joints的目标位置（使用物理驱动，避免穿模）"""
        from isaacsim.core.utils.types import ArticulationActions
        if self._articulation is None:
            raise RuntimeError("Articulation uninitialized")

        if not isinstance(target_arm_positions, torch.Tensor):
            target_arm_positions = torch.tensor(target_arm_positions, dtype=torch.float32)

        if target_arm_positions.shape[0] != 14:
            raise ValueError(f"Expected14joint positions，got {target_arm_positions.shape[0]} ")

        if target_arm_positions.ndim == 1:
            target_arm_positions = target_arm_positions.unsqueeze(0)

        joint_indices = torch.tensor(self.arm_joint_indices, dtype=torch.int32)
        
        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=target_arm_positions,
                joint_indices=joint_indices
            )
        )

    def set_body_joint_positions(self, target_body_positions, task_num: int = None):
        """设置除 arm/finger 外所有关节位置。

        允许输入:
        - 标量: 自动广播到所有 body joints
        - 序列/张量: 长度必须等于 body joints 数量
        """
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            raise RuntimeError("Articulation uninitialized")

        if not self.body_joint_indices:
            return

        if isinstance(target_body_positions, (int, float)):
            target_body_positions = [float(target_body_positions)] * len(self.body_joint_indices)

        if not isinstance(target_body_positions, torch.Tensor):
            target_body_positions = torch.tensor(target_body_positions, dtype=torch.float32)

        target_body_positions = target_body_positions.flatten()
        if target_body_positions.shape[0] != len(self.body_joint_indices):
            raise ValueError(
                f"Body joints expected {len(self.body_joint_indices)} positions, got {target_body_positions.shape[0]}"
            )

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=target_body_positions.unsqueeze(0),
                joint_indices=torch.tensor(self.body_joint_indices, dtype=torch.int32),
            )
        )
    def set_finger_positions(self, target_fingers, side: Optional[str] = None, task_num: int = None):
        """Set gripper target positions.

        side semantics:
        - None: expects 4D [L_f1, L_f2, R_f1, R_f2]
        - "left"/"right": expects 2D

        NaN in target_fingers means "skip position control for that joint" (effort-only mode).
        """
        import math
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            raise RuntimeError("Articulation uninitialized")

        if not isinstance(target_fingers, torch.Tensor):
            target_fingers = torch.tensor(target_fingers, dtype=torch.float32)
        target_fingers = target_fingers.flatten()

        if side == "left":
            if target_fingers.shape[0] != 2:
                raise ValueError(f"left GripperExpected2joint positions，got {target_fingers.shape[0]} ")
            all_indices = self.finger_joint_indices[:2]
        elif side == "right":
            if target_fingers.shape[0] != 2:
                raise ValueError(f"right GripperExpected2joint positions，got {target_fingers.shape[0]} ")
            all_indices = self.finger_joint_indices[2:4]
        else:
            if target_fingers.shape[0] != 4:
                raise ValueError(f"机器人接口Expected4Finger joint位置，got {target_fingers.shape[0]} ")
            all_indices = self.finger_joint_indices

        # Filter out NaN: NaN means effort-only for that joint, skip position drive
        valid_pos = []
        valid_idx = []
        for pos, idx in zip(target_fingers.tolist(), all_indices):
            if not math.isnan(pos):
                valid_pos.append(pos)
                valid_idx.append(idx)

        if not valid_pos:
            return

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([valid_pos], dtype=torch.float32),
                joint_indices=torch.tensor(valid_idx, dtype=torch.int32),
            )
        )

    def move_home(self):
        """将机器人重置到初始化时的关节初始Pose，并同步重置IK相关状态。

        Reset includes:
        1. All joint positions return to initial values set by ``initialize()``
        2. Zero out all joint velocities when possible
        3. 清空IK平滑缓存、告警计数和累计时间
        4. 若IK已初始化，则重建其warm-start初始参考
        """
        initial_positions = np.asarray(self.initial_joint_positions, dtype=np.float32)
        all_indices = np.asarray(self._all_dof_indices, dtype=np.int32)
        self._articulation.set_joint_positions(
            torch.tensor(initial_positions, dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(all_indices), dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self.time = 0.0
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            self.ik_solver.sync_joint_positions(
                self._articulation.dof_names,
                initial_positions.tolist(),
            )
            self.ik_solver.save_initial_q()

    def apply_finger_efforts(self, efforts: list[float]) -> None:
        """对Finger joint施加力矩（4D: [L_f1, L_f2, R_f1, R_f2]）。"""
        from isaacsim.core.utils.types import ArticulationActions
        if self._articulation is None:
            return
        t = torch.tensor([efforts], dtype=torch.float32)
        idx = torch.tensor(self.finger_joint_indices, dtype=torch.int32)
        self._articulation.apply_action(
            ArticulationActions(joint_efforts=t, joint_indices=idx)
        )

    def close_gripper(self, side: Optional[str] = None, task_name: Optional[str] = None):
        """Close gripper on specified side (without world.step, safe to call in callback)"""
        from isaacsim.core.utils.types import ArticulationActions
        target_pos = [self.gripper_close_width] * 2

        if side == "left":
            control_finger_indices = torch.tensor(self.finger_joint_indices[:2], dtype=torch.int32)
        elif side == "right":
            control_finger_indices = torch.tensor(self.finger_joint_indices[2:4], dtype=torch.int32)
        else:
            control_finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.int32)
            target_pos = target_pos * 2

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([target_pos], dtype=torch.float32),
                joint_indices=control_finger_indices,
            )
        )

    def open_gripper(self, side: Optional[str] = None, task_name: Optional[str] = None):
        """Open gripper on specified side (without world.step, safe to call in callback)"""
        from isaacsim.core.utils.types import ArticulationActions
        target_pos = [self.gripper_open_width] * 2

        if side == "left":
            control_finger_indices = torch.tensor(self.finger_joint_indices[:2], dtype=torch.int32)
        elif side == "right":
            control_finger_indices = torch.tensor(self.finger_joint_indices[2:4], dtype=torch.int32)
        else:
            control_finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.int32)
            target_pos = target_pos * 2

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([target_pos], dtype=torch.float32),
                joint_indices=control_finger_indices,
            )
        )

    def get_camera_rgb(self, camera_name):
        """Get camera RGB"""
        if camera_name not in self.cameras:
            return None
        try:
            return self.cameras[camera_name].get_rgb()
        except Exception:
            return None

    def get_camera_depth(self, camera_name):
        """Get camera depth"""
        if camera_name not in self.cameras:
            return None
        try:
            return self.cameras[camera_name].get_depth()
        except Exception:
            return None

    def get_camera_rgbd(self, camera_name):
        """Get camera RGB-D"""
        return {
            'rgb': self.get_camera_rgb(camera_name),
            'depth': self.get_camera_depth(camera_name),
            'camera_name': camera_name,
        }

    def get_sixforce(self):
        wrench_data_list = []
        sensor_joint_forces = self._articulation.get_measured_joint_forces()[0]

        for joint_name in self.sixforce_joint_names:
            joint_index = self._articulation.get_joint_index(joint_name)
            sixforce_data = sensor_joint_forces[joint_index + 1]
            wrench_data = {
                'frame_id': joint_name.replace("_joint", "_frame"),
                'force': sixforce_data[:3].tolist(),
                'torque': sixforce_data[3:].tolist()
            }
            wrench_data_list.append(wrench_data)
        return wrench_data_list

    def initialize_ik(self, urdf_path: str):
        """Initialize Pinocchio IK solver and build joint index mapping"""
        from Ubtech_sim.source.DualArmIK import DualArmIK
        logger.info(f"Initializing DualArmIK, using URDF: {urdf_path}")
        self.ik_solver = DualArmIK(urdf_path)
        logger.info("DualArmIK initialized successfully, building joint index mapping...")

        dof_names = self._articulation.dof_names
        self._left_arm_isaac_indices = []
        for jname in DualArmIK.LEFT_ARM_JOINTS:
            if jname in dof_names:
                self._left_arm_isaac_indices.append(self._articulation.get_dof_index(jname))

        self._right_arm_isaac_indices = []
        for jname in DualArmIK.RIGHT_ARM_JOINTS:
            if jname in dof_names:
                self._right_arm_isaac_indices.append(self._articulation.get_dof_index(jname))

        WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint"]
        self._waist_isaac_indices = []
        self._waist_init_positions = []
        for jname in WAIST_JOINTS:
            if jname == 'waist_pitch_joint':
                idx = self._articulation.get_dof_index(jname)
                self._waist_isaac_indices.append(idx)
                self._waist_init_positions.append(float(-0.0))
            elif jname in dof_names:
                idx = self._articulation.get_dof_index(jname)
                self._waist_isaac_indices.append(idx)
                self._waist_init_positions.append(float(0.0))

        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        joints = self.get_joint_states()
        if joints is not None:
            self.ik_solver.sync_joint_positions(
                joints['all_names'], joints['all_positions']
            )

        self.ik_solver.save_initial_q()

        left_neutral = [self._joint_value_map.get(j, 0.0) for j in DualArmIK.LEFT_ARM_JOINTS]
        right_neutral = [self._joint_value_map.get(j, 0.0) for j in DualArmIK.RIGHT_ARM_JOINTS]
        self.ik_solver.set_neutral_config(left_neutral, right_neutral)

        print(f"[DualArmIK] Initialization complete  Left arm {len(self._left_arm_isaac_indices)} DOF, "
              f"Right arm {len(self._right_arm_isaac_indices)} DOF, "
              f"Waist locked {len(self._waist_isaac_indices)} DOF")

    def get_ee_poses(self):
        """Get current dual-arm end-effector xyzrpy (first sync Isaac Sim joint states to pinocchio)"""
        joints = self.get_joint_states()
        if joints is None or self.ik_solver is None:
            return None
        self.ik_solver.sync_joint_positions(joints['all_names'], joints['all_positions'])
        return self.ik_solver.get_both_ee_poses()

    def control_dual_arm_ik(
        self,
        step_size: float,
        left_target_xyzrpy=None,
        right_target_xyzrpy=None,
        **ik_kwargs,
    ):
        """Dual-arm IK control: Send left and right arm end-effector targets [x,y,z,roll,pitch,yaw] simultaneously.

        Args:
            step_size: Physics step size (seconds)
            left_target_xyzrpy: Left arm target pose, None keeps current position
            right_target_xyzrpy: Right arm target pose, None keeps current position
            **ik_kwargs: Parameters passed to IK solver (max_iter, pos_tol, rot_tol, damping, dt)
        """
        from isaacsim.core.utils.types import ArticulationActions
        self.time += step_size

        if self.ik_solver is None:
            print("[DualArmIK] IK solver not initialized, please call initialize_ik() first")
            return

        joints = self.get_joint_states()
        if joints is None:
            return

        isaac_names = joints['all_names']
        isaac_positions = joints['all_positions']

        ik_result = self.ik_solver.solve_dual_arm(
            left_target_xyzrpy=left_target_xyzrpy,
            right_target_xyzrpy=right_target_xyzrpy,
            isaac_joint_names=isaac_names,
            isaac_joint_positions=isaac_positions,
            **ik_kwargs,
        )

        all_indices = []
        all_positions = []
        warn_msg = []

        if 'left_joint_positions' in ik_result:
            smoothed = self._smooth_joints('left', ik_result['left_joint_positions'])
            all_indices.extend(self._left_arm_isaac_indices)
            all_positions.extend(smoothed.tolist())
            if not ik_result['left_success']:
                warn_msg.append("left arm")

        if 'right_joint_positions' in ik_result:
            smoothed = self._smooth_joints('right', ik_result['right_joint_positions'])
            all_indices.extend(self._right_arm_isaac_indices)
            all_positions.extend(smoothed.tolist())
            if not ik_result['right_success']:
                warn_msg.append("right arm")

        if warn_msg:
            self._ik_warn_counter += 1
            if self._ik_warn_counter >= 200:
                diag = ""
                if hasattr(self.ik_solver, '_last_fail_info'):
                    info = self.ik_solver._last_fail_info
                    diag = (f" | pos_err={info['pos_err']:.4f}m"
                            f" rot_err={info['rot_err']:.4f}rad"
                            f" rot_tol={info['effective_rot_tol']:.4f}")
                print(f"[DualArmIK] Warning: {', '.join(warn_msg)} IK not converged{diag}")
                self._ik_warn_counter = 0
        else:
            self._ik_warn_counter = 0

        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        if len(all_indices) > 0:
            self._articulation.apply_action(
                ArticulationActions(
                    joint_positions=torch.tensor([all_positions], dtype=torch.float32),
                    joint_indices=torch.tensor(all_indices, dtype=torch.int32),
                )
            )
        ik_result['smoothed_positions'] = all_positions
        return ik_result

    def _smooth_joints(self, side: str, ik_positions: np.ndarray) -> np.ndarray:
        """EMA smoothing of IK joint output to reduce jitter."""
        ik_positions = np.asarray(ik_positions, dtype=float)
        if side not in self._last_arm_positions:
            self._last_arm_positions[side] = ik_positions.copy()
            return ik_positions

        prev = self._last_arm_positions[side]
        alpha = self._smooth_alpha
        smoothed = prev + alpha * (ik_positions - prev)
        self._last_arm_positions[side] = smoothed.copy()
        return smoothed

