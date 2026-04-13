import os
import random
from typing_extensions import override
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.prims import Articulation, XFormPrim
import numpy as np
from isaacsim.core.cloner import Cloner
from functools import partial
import omni.replicator.core as rep
from isaacsim.core.prims import SingleRigidPrim, RigidPrim, Articulation
from isaacsim.core.simulation_manager import SimulationManager
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class SceneBuilder:
    """Configurable scene builder driven by a YAML config dict."""

    def __init__(self, cfg, data_logger):
        self.cfg = cfg
        self.root_path = cfg['root_path']
        # created object references
        self.table = None
        self.plane = None
        self.multi_parts = None
        self.box = None
        self.foam = None
        self.robot = None
        self.robot_prim_path = None
        
        # 从配置中获取目标物体列表，默认为空列表
        self.target_objects = cfg.get('target_objects', [])
        # 存储各物体的prim路径
        self.table_prim_paths = []
        self.box_prim_paths = []
        self.parts_prim_paths = []
        self.pose_logger = data_logger
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # SimulationManager.initialize_physics()

    def _usd_path(self, relative):
        return os.path.join(self.root_path, relative)

    def build_table(self):
        self.table_cfg = self.cfg['table']
        self.table = rep.create.from_usd(
            self._usd_path(self.table_cfg['table_usd']),
            position=rep.distribution.choice(self.table_cfg['table_position'], with_replacements=False),
            scale=rep.distribution.choice(self.table_cfg['table_scale'], with_replacements=False),
            count=len(self.table_cfg['table_position']),
        )
        # 保存prim路径
        if self.table is not None:
            prims_info = self.table._get_prims()
            if 'primsIn' in prims_info:
                prims_in = prims_info['primsIn']
                if not isinstance(prims_in, (list, tuple)):
                    prims_in = [prims_in]
                self.table_prim_paths = [str(prim.GetPath()) + "/Ref/material" for prim in prims_in]
        return self.table

    def build_ConveyorBelt(self):
        self.ConveyorBelt_cfg = self.cfg['ConveyorBelt']
        self.ConveyorBelt = rep.create.from_usd(
            self._usd_path(self.ConveyorBelt_cfg['ConveyorBelt_usd']),
            position=self.ConveyorBelt_cfg['ConveyorBelt_position'],
            scale=self.ConveyorBelt_cfg['ConveyorBelt_scale'],
        )
        # 保存传送带初始位置，用于重置
        self._conveyor_initial_position = self.ConveyorBelt_cfg['ConveyorBelt_position']
        return self.ConveyorBelt

    def build_parts(self):
        self.plane_cfg = self.cfg['plane']
        self.part_cfg = self.cfg['part']
        num_planes = len(self.plane_cfg['plane_position'])
        self.planes = []
        for i in range(num_planes):
            self.planes.append(rep.create.plane(
                position=self.plane_cfg['plane_position'][i],
                scale=self.plane_cfg['plane_scale'][i],
                visible=False,
            ))

        part_usds = [self._usd_path(v) for k, v in self.part_cfg.items()
                     if k.endswith('_usd')]

        self.parts_list = []

        if self.cfg['task_number'] == 3:
            for i in range(len(self.box_cfg['box_position'])):
            # 使用索引 i 同时指定当前的零件和对应的平面
            # 箱子 0 对应零件 A，箱子 1 对应零件 B
                current_part_usd = part_usds[i] 
                target_plane = self.planes[i]

            # 每次生成 num_parts 个该种类的工件
                new_parts = rep.create.from_usd(
                usd=current_part_usd,
                count=self.part_cfg.get('num_parts', 3),
                semantics={"class": "part"}
                )

                self.parts_list.append(new_parts)

                with new_parts:
                    rep.physics.rigid_body(overwrite=True)
                    rep.physics.mass(mass=0.2)

                    # 随机旋转
                    rep.modify.pose(
                        rotation=rep.distribution.uniform((-90, -90, -90), (90, 90, 90))
                    )

                    # 散布到对应的箱子平面上
                    rep.randomizer.scatter_2d(
                        surface_prims=target_plane,
                        check_for_collisions=True,
                    )            

            # 保存零件 prim 路径
            self._extract_parts_prim_paths()

        elif self.cfg['task_number'] == 2:
            from isaacsim.core.prims import RigidPrim

            num_parts = self.part_cfg.get('num_parts', 4)

            # 源 prim 直接命名为 _0，与 generate_paths 输出对齐
            self.part_A = stage_utils.add_reference_to_stage(
                usd_path=part_usds[0],
                prim_path='/Root/Part_A_0',
            )
            self.part_B = stage_utils.add_reference_to_stage(
                usd_path=part_usds[1],
                prim_path='/Root/Part_B_0',
            )

            self.cloner = Cloner()
            # generate_paths("/Root/Part_A", num_parts) → [_0, _1, ..., _{num_parts-1}]
            # 含源 prim _0，克隆体以 USD Inherits 继承源，物理状态各自独立
            target_paths_A = self.cloner.generate_paths("/Root/Part_A", num_parts)
            self.cloner.clone(
                source_prim_path="/Root/Part_A_0",
                prim_paths=target_paths_A,
            )

            target_paths_B = self.cloner.generate_paths("/Root/Part_B", num_parts)
            self.cloner.clone(
                source_prim_path="/Root/Part_B_0",
                prim_paths=target_paths_B,
            )

            clone_paths = list(target_paths_A) + list(target_paths_B)

            # RigidPrim 使用精确路径列表，总数 = num_parts * 2
            self.rigid_prim = RigidPrim(
                prim_paths_expr=clone_paths,
                name="rigid_prim_view"
            )

            total_parts = num_parts * 2
            random_indices = np.random.permutation(np.arange(total_parts))
            start_position = -0.3 - total_parts * self.part_cfg['part_distance']
            init_positions = np.column_stack([
                np.linspace(start_position, -0.3, total_parts),
                np.full(total_parts, 0.278),
                np.full(total_parts, 0.98),
            ])

            self.rigid_prim.set_world_poses(
                positions=init_positions,
                indices=random_indices
            )

            # 保存 Task2 初始位姿，用于 scatter_after_reset 和 reset
            self._task2_initial_positions = init_positions.copy()
            self._task2_initial_indices = random_indices.copy()

            # 保存零件 prim 路径，用于 get_parts_world_poses 和 save_parts_poses
            self.parts_prim_paths = clone_paths
            print(f"[SceneBuilder] Task2 初始化: 发现 {len(self.parts_prim_paths)} 个零件")


        elif self.cfg['task_number'] == 1:
            part_a_pool = self.part_cfg.get('part_a_assets', [])
            part_b_pool = self.part_cfg.get('part_b_assets', [])
            num_to_create = self.part_cfg.get('num_parts', 2) # 每种创建几个

            self.parts_list = []

            # 1. 为 Part A 随机选择资产并创建
            for i in range(num_to_create):
                chosen_a = self._usd_path(random.choice(part_a_pool))
                self.parts_list.append(rep.create.from_usd(
                    usd=chosen_a,
                    count=1, # 每次创建一个，确保独立随机
                    semantics={"class": "part_a"}
                ))

            # 2. 为 Part B 随机选择资产并创建
            for i in range(num_to_create):
                chosen_b = self._usd_path(random.choice(part_b_pool))
                self.parts_list.append(rep.create.from_usd(
                    usd=chosen_b,
                    count=1,
                    semantics={"class": "part_b"}
                ))

            # 3. 统一处理物理和随机位置（保持原有逻辑）
            self.parts_group = rep.create.group(items=self.parts_list)

            with self.parts_group:
                rep.physics.rigid_body(overwrite=True)
                rep.physics.mass(mass=self.part_cfg.get('mass', 0.2))
                rep.modify.pose(
                    rotation=rep.distribution.uniform((-90, -90, -90), (90, 90, 90))
                )
                rep.randomizer.scatter_2d(
                    surface_prims=self.planes[0],
                    check_for_collisions=True
                )
            # 保存零件 prim 路径（仅 Task1）
            self._extract_parts_prim_paths()

    def _extract_parts_prim_paths(self):
        """从 replicator 节点中提取零件的 USD prim 路径。

        流程（对齐 NVIDIA 官方推荐模式）：
        1. Replicator 负责创建物体（build_parts）
        2. 此处仅提取路径，不创建 SingleRigidPrim
        3. SingleRigidPrim 延迟到首次使用时创建（world.reset() 之后），
           避免在 world.reset() 期间 Replicator 重建 prim 导致 tensor view 失效
        """
        self.parts_prim_paths = []
        self._rigid_body_paths = []
        self._parts_rigid_prims = []
        self._rigid_prims_initialized = False

        for part_rep in self.parts_list:
            try:
                prims_info = part_rep._get_prims()
                if 'primsIn' in prims_info:
                    prims_in = prims_info['primsIn']
                    if not isinstance(prims_in, (list, tuple)):
                        prims_in = [prims_in]
                    for prim in prims_in:
                        path = str(prim.GetPath())
                        self.parts_prim_paths.append(path)
            except Exception as e:
                print(f"[SceneBuilder] 提取零件路径失败: {e}")

        # 永久保存初始路径，重置时复用
        self._initial_parts_prim_paths = list(self.parts_prim_paths)
        print(f"[SceneBuilder] 提取到 {len(self.parts_prim_paths)} 个零件路径: {self.parts_prim_paths}")

    def _ensure_rigid_prims(self):
        """确保 SingleRigidPrim 缓存已创建（延迟初始化）。

        必须在 world.reset() 之后调用，否则 Replicator 重建 prim 会导致 tensor view 失效。
        """
        if getattr(self, '_rigid_prims_initialized', False):
            return
        self._rebuild_rigid_prims(self.parts_prim_paths)

    def _rebuild_rigid_prims(self, prim_paths: list):
        """根据给定路径列表，查找 RigidBodyAPI 并缓存 SingleRigidPrim。

        统一被 _ensure_rigid_prims / _randomize_task1_assets 调用。
        """
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()

        self._rigid_body_paths = []
        self._parts_rigid_prims = []

        for path in prim_paths:
            rb_path = self._find_rigid_body_path(stage, path)
            self._rigid_body_paths.append(rb_path)
            if rb_path is not None:
                try:
                    self._parts_rigid_prims.append(SingleRigidPrim(prim_path=rb_path))
                except Exception as e:
                    print(f"[SceneBuilder] 创建 SingleRigidPrim 失败 {rb_path}: {e}")
                    self._parts_rigid_prims.append(None)
            else:
                self._parts_rigid_prims.append(None)

        self._rigid_prims_initialized = True
        ok = sum(1 for r in self._parts_rigid_prims if r is not None)
        print(f"[SceneBuilder] 发现 {len(prim_paths)} 个零件 ({ok} 个有 rigid body): {prim_paths}")

    @staticmethod
    def _find_rigid_body_path(stage, base_path: str):
        """在 base_path 及其直接子 prim 中查找带 UsdPhysics.RigidBodyAPI 的路径。

        返回找到的路径，找不到返回 None。
        """
        from pxr import UsdPhysics
        prim = stage.GetPrimAtPath(base_path)
        if not prim.IsValid():
            return None
        # 优先检查自身
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return base_path
        # 检查直接子 prim
        for child in prim.GetChildren():
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(child.GetPath())
        return None

    def get_parts_world_poses(self):
        """查询所有零件当前的世界坐标位姿 (通过 USD XformCache)

        Returns:
            list of dict: [{'prim_path': str,
                            'position': [x, y, z],
                            'orientation': [qx, qy, qz, qw]}, ...]
        """
        from pxr import UsdGeom, Usd
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        # Create fresh XformCache with current time to get updated transforms
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        results = []
        for prim_path in self.parts_prim_paths:
            try:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    print(f"[SceneBuilder] prim 无效: {prim_path}")
                    continue
                world_tf = xform_cache.GetLocalToWorldTransform(prim)
                t = world_tf.ExtractTranslation()
                q = world_tf.ExtractRotationQuat()
                qi = q.GetImaginary()
                qr = q.GetReal()
                pos = [float(t[0]), float(t[1]), float(t[2])]
                results.append({
                    'prim_path': prim_path,
                    'position': pos,
                    'orientation': [float(qi[0]), float(qi[1]), float(qi[2]), float(qr)],
                })
                # Debug: print if position is still [0,0,0]
                if abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6 and abs(pos[2]) < 1e-6:
                    print(f"[SceneBuilder] 警告: {prim_path} 位置仍为 [0,0,0]，可能是容器节点而非实际物体")
            except Exception as e:
                print(f"[SceneBuilder] 查询 {prim_path} 位姿失败: {e}")
        return results

    # ===================== 统一物体位姿接口（供数据采集/回放使用） =====================

    @staticmethod
    def compute_num_tracked_objects(task_cfg: dict) -> int:
        """根据任务配置计算被追踪物体的数量（不需要场景实例，可在 connect 前调用）"""
        task = task_cfg.get('task_number', 0)
        if task == 1:
            return task_cfg.get('part', {}).get('num_parts', 2) * 2
        elif task == 2:
            return task_cfg.get('part', {}).get('num_parts', 4) * 2
        elif task == 3:
            num_boxes = len(task_cfg.get('box', {}).get('box_position', []))
            num_parts = task_cfg.get('part', {}).get('num_parts', 3)
            return num_boxes * num_parts
        elif task == 4:
            return 0  # 箱子为静态 XFormPrim，位置固定，无需记录
        return 0

    def get_object_poses_flat(self) -> np.ndarray:
        """返回所有被追踪物体的位姿，展平为一维数组。

        每个物体 7 个值: [x, y, z, qx, qy, qz, qw]
        总维度 = num_objects * 7

        Returns:
            np.ndarray: shape (num_objects * 7,), dtype float32
        """
        task = self.cfg.get('task_number', 0)

        if task in (1, 3):
            poses = self.get_parts_world_poses()
            result = []
            for p in poses:
                result.extend(p['position'])       # [x, y, z]
                result.extend(p['orientation'])     # [qx, qy, qz, qw]
            return np.array(result, dtype=np.float32)

        elif task == 2:
            if not hasattr(self, 'rigid_prim') or self.rigid_prim is None:
                n = self.compute_num_tracked_objects(self.cfg)
                return np.zeros(n * 7, dtype=np.float32)
            positions, orientations = self.rigid_prim.get_world_poses()
            # Isaac Sim 返回 wxyz，转换为 xyzw 与 Task1/3 一致
            result = []
            for i in range(positions.shape[0]):
                result.extend(positions[i].tolist())
                w, x, y, z = orientations[i].tolist()
                result.extend([x, y, z, w])
            return np.array(result, dtype=np.float32)

        elif task == 4:
            if not hasattr(self, 'box_articulation') or self.box_articulation is None:
                return np.zeros(7, dtype=np.float32)
            pos, ori = self.box_articulation.get_world_poses()
            w, x, y, z = ori[0].tolist()
            result = list(pos[0].tolist()) + [x, y, z, w]
            return np.array(result, dtype=np.float32)

        return np.array([], dtype=np.float32)

    def set_object_poses_from_flat(self, flat_poses: np.ndarray) -> None:
        """从展平的位姿数组恢复物体位置（用于 replay 初始化场景）。

        Args:
            flat_poses: shape (num_objects * 7,)，格式 [x,y,z,qx,qy,qz,qw, ...]
        """
        from pxr import UsdGeom, Gf
        import omni.usd

        task = self.cfg.get('task_number', 0)
        num_objects = self.compute_num_tracked_objects(self.cfg)
        if flat_poses.shape[0] != num_objects * 7:
            print(f"[SceneBuilder] set_object_poses_from_flat: 维度不匹配 "
                  f"(期望 {num_objects * 7}, 得到 {flat_poses.shape[0]})")
            return

        poses_7 = flat_poses.reshape(num_objects, 7)  # (N, 7)

        if task in (1, 3):
            # 延迟初始化 SingleRigidPrim（确保在 world.reset() 之后）
            self._ensure_rigid_prims()
            rigid_prims = getattr(self, '_parts_rigid_prims', [])
            count = min(num_objects, len(self.parts_prim_paths))
            positions = poses_7[:count, :3].astype(np.float64)
            # xyzw → wxyz
            quats = poses_7[:count, 3:].astype(np.float64)
            orientations = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])

            restored = 0
            for i in range(count):
                if i < len(rigid_prims) and rigid_prims[i] is not None:
                    rigid_prims[i].set_world_pose(
                        position=positions[i], orientation=orientations[i])
                    rigid_prims[i].set_linear_velocity(np.zeros(3))
                    rigid_prims[i].set_angular_velocity(np.zeros(3))
                    restored += 1
                else:
                    print(f"[SceneBuilder] set_object_poses: 无 rigid body {self.parts_prim_paths[i]}")
            print(f"[SceneBuilder] 已从 flat 数据恢复 {restored}/{count} 个物体位姿")

        elif task == 2:
            if not hasattr(self, 'rigid_prim') or self.rigid_prim is None:
                return
            positions = poses_7[:, :3].astype(np.float64)
            # xyzw → wxyz
            quats = poses_7[:, 3:].astype(np.float64)
            orientations = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])
            self.rigid_prim.set_world_poses(positions=positions, orientations=orientations)
            self.rigid_prim.set_velocities(velocities=np.zeros((num_objects, 6)))

        elif task == 4:
            if not hasattr(self, 'box_articulation') or self.box_articulation is None:
                return
            x, y, z, qx, qy, qz, qw = poses_7[0]
            pos = np.array([[x, y, z]], dtype=np.float64)
            ori = np.array([[qw, qx, qy, qz]], dtype=np.float64)  # wxyz
            self.box_articulation.set_world_poses(positions=pos, orientations=ori)
            if hasattr(self.box_articulation, 'set_velocities'):
                self.box_articulation.set_velocities(velocities=np.zeros((1, 6)))

        print(f"[SceneBuilder] 已从 flat 数据恢复 {num_objects} 个物体位姿")

    # Deprecated: 接口没实现功能，需要reset后保存parts的world poses。 replay时直接读取JSON文件，将parts放置到指定位姿。 TODO
    def set_parts_world_poses(self, json_file_path: Optional[str] = None) -> None:
        """
        从JSON文件读取位姿配置，使用 Isaac Sim / USD 官方标准接口设置世界位姿
        
        JSON 格式要求：
        - prim_path: 物体路径
        - position: [x, y, z]
        - orientation: [x, y, z, w]  (XYZW顺序，和你的get接口一致)
        """
        repo_root = Path(__file__).resolve().parent.parent
        if json_file_path is None:
            pose_path = repo_root / "task1_parts_poses_20260331_161915.json"
        else:
            p = Path(json_file_path).expanduser()
            pose_path = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
        # 读取JSON
        with open(pose_path, 'r', encoding='utf-8') as f:
            pose_configs = json.load(f)
        from pxr import Usd, UsdGeom, Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        for cfg in pose_configs:
            prim_path = cfg["prim_path"]
            pos = cfg["position"]
            orient = cfg["orientation"]  # 输入顺序：x y z w

            # 获取prim
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue
            if not prim.IsA(UsdGeom.Xformable):
                continue

            xformable = UsdGeom.Xformable(prim)

            # ======================= 官方唯一正确方式 =======================
            # 1. 清空所有变换（官方标准）
            xformable.ClearXformOpOrder()

            # 2. 四元数转换 XYZW → WXYZ
            x_q, y_q, z_q, w_q = orient
            quat = Gf.Quatd(w_q, x_q, y_q, z_q)

            # 3. 构建变换矩阵
            mat = Gf.Matrix4d()
            mat.SetTranslate(Gf.Vec3d(*pos))
            mat.SetRotate(quat)

            # 4. 添加变换操作并设置矩阵（全版本支持）
            xform_op = xformable.AddTransformOp()
            xform_op.Set(mat)
            # =================================================================

            print(f"✅ 已设置 {prim_path} 世界位姿")



    def build_box(self):
        self.box_cfg = self.cfg['box']
        num_boxes = len(self.box_cfg['box_position'])

        self.box = stage_utils.add_reference_to_stage(
            usd_path=self._usd_path(self.box_cfg['box_usd']),
            prim_path='/Root/Box',
        )

        if self.cfg['task_number'] <= 3:
            from isaacsim.core.cloner import Cloner
            self.cloner = Cloner()
            target_paths = self.cloner.generate_paths("/Root/Box", num_boxes - 1)
            self.cloner.clone(
                source_prim_path="/Root/Box",
                prim_paths=target_paths
            )

            self.boxes = XFormPrim(
                prim_paths_expr='/Root/Box.*',
                positions=np.array(self.box_cfg['box_position']),
                scales=np.array(self.box_cfg['box_scale']),
            )

            # 箱子位置锁定：将刺有物理属性的箱子设为 kinematic
            if self.box_cfg.get('lock_boxes', False):
                self._lock_box_positions()

        elif self.cfg['task_number'] == 4:
            self.box_articulation = Articulation(
                prim_paths_expr='/Root/Box',
                positions=np.array([
                    self.box_cfg['box_position'],
                ]),
                scales=np.array([
                    self.box_cfg['box_scale'],
                ]),
                name='Box',
            )
            self.box_initial_joint_positions = self.box_articulation.get_joint_positions()
            self._box_initial_world_pos, self._box_initial_world_ori = self.box_articulation.get_world_poses()

        return self.box

    def _lock_box_positions(self):
        """将 /Root/Box* 下所有 RigidBody prim 设为 kinematic，禁止物理引擎移动箱子。"""
        try:
            import omni.usd
            from pxr import UsdPhysics, Usd
            stage = omni.usd.get_context().get_stage()
            locked = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith("/Root/Box"):
                    continue
                if UsdPhysics.RigidBodyAPI.CanApply(prim):
                    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                    rb_api.CreateKinematicEnabledAttr(True)
                    locked += 1
                elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb_api = UsdPhysics.RigidBodyAPI(prim)
                    rb_api.CreateKinematicEnabledAttr(True)
                    locked += 1
            print(f"[SceneBuilder] 箱子锁定完成: {locked} 个 RigidBody 已设为 kinematic")
        except Exception as e:
            print(f"[SceneBuilder] 箱子锁定失败: {e}")

    def _apply_robot_pose(self):
        """将 YAML 中 robot_position / robot_rotation 写入机器人 USD 内的
        world-anchored PhysicsFixedJoint（localPos0 / localRot0）。

        rep.create.from_usd 只设置 XForm 变换；物理引擎启动后
        FixedJoint 会用 USD 内的硬编码值覆盖该变换。
        此方法在物理启动前将关节目标坐标改写为 YAML 值，从而让
        robot_position / robot_rotation 真正生效。
        """
        import omni.usd
        import math
        from pxr import Gf, Usd

        robot_cfg = self.cfg['robot']
        pos = robot_cfg['robot_position']                    # [x, y, z]  metres
        rot_deg = robot_cfg.get('robot_rotation', [0, 0, 0]) # [roll, pitch, yaw] degrees ZYX

        # Euler ZYX (degrees) → unit quaternion (w, x, y, z)
        r_rad, p_rad, y_rad = (math.radians(a) for a in rot_deg)
        cy, sy = math.cos(y_rad * 0.5), math.sin(y_rad * 0.5)
        cp, sp = math.cos(p_rad * 0.5), math.sin(p_rad * 0.5)
        cr, sr = math.cos(r_rad * 0.5), math.sin(r_rad * 0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        
        stage = omni.usd.get_context().get_stage()
        n_updated = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "PhysicsFixedJoint":
                continue
            # 只处理 world-anchored 关节 (body0 无目标 = 锚定到世界)
            body0_rel = prim.GetRelationship("physics:body0")
            if body0_rel.IsValid() and body0_rel.GetTargets():
                continue
            # 覆盖世界侧附着点
            pos_attr = prim.GetAttribute("physics:localPos0")
            rot_attr = prim.GetAttribute("physics:localRot0")
            if pos_attr.IsValid():
                pos_attr.Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
            if rot_attr.IsValid():
                rot_attr.Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
            print(f"[SceneBuilder] FixedJoint 位姿已更新: {prim.GetPath()}  "
                  f"pos={[round(v, 4) for v in pos]}  rot_deg={rot_deg}")
            n_updated += 1

        if n_updated == 0:
            print("[SceneBuilder] 警告: 未找到 world-anchored PhysicsFixedJoint，"
                  "robot_position/rotation 可能无法通过 YAML 生效")
        else:
            print(f"[SceneBuilder] 已覆盖 {n_updated} 个 FixedJoint — "
                  f"robot 将在 YAML 指定位置 {pos} 生成")

    def build_robot(self):
        """构建机器人并返回prim路径"""
        robot_cfg = self.cfg['robot']
        self.robot = rep.create.from_usd(
            self._usd_path(robot_cfg['robot_usd']),
            position=robot_cfg['robot_position'],
            rotation=robot_cfg['robot_rotation'],
            parent='/Root',
        )
        if self.robot is not None:
            prims_info = self.robot._get_prims()
            if 'primsIn' in prims_info and prims_info['primsIn']:
                self.robot_prim_path = str(prims_info['primsIn'][0].GetPath())
        # 将 YAML robot_position/rotation 写入 FixedJoint，使其在物理启动后生效
        self._apply_robot_pose()
        return self.robot

    def build_foam(self):
        foam_cfg = self.cfg['foam']
        foam_position = foam_cfg.get('foam_position', foam_cfg.get('robot_position'))
        if foam_position is None:
            raise KeyError("foam 配置缺少 foam_position（或兼容字段 robot_position）")

        if self.cfg.get('task_number') == 4:
            # 任务4: 箱子资产自带 foam mesh，不需要独立创建
            print(f"[SceneBuilder] Task4 箱子资产已自带 foam mesh，跳过独立创建")
        else:
            self.foam = rep.create.from_usd(
                self._usd_path(foam_cfg['foam_usd']),
                position=foam_position,
                rotation=foam_cfg.get('foam_rotation', [0, 0, 0]),
                scale=foam_cfg.get('foam_scale', [1, 1, 1]),
                parent='/Replicator',
            )
        return self.foam

    def sync_foam_to_box(self):
        """任务4: 箱子资产已自带 foam mesh，无需同步（保留接口兼容）。"""
        pass

    def build_all(self):
        """Build every object defined in the config."""
        if "table" in self.cfg.keys():
            self.build_table()
        if "ConveyorBelt" in self.cfg.keys():
            self.build_ConveyorBelt()
        if "box" in self.cfg.keys():
            self.build_box()
        if "foam" in self.cfg.keys():
            self.build_foam()
        if "part" in self.cfg.keys():
            self.build_parts()


    def setup_target_objects_rigidbody(self):
        """Setup rigidbody properties for target objects and get their world poses."""
        
        world_poses = {}
        # Return empty if no target_objects defined in config
        if not self.target_objects:
            return world_poses
        
        # Collect prim paths from target_objects configuration
        for target_obj in self.target_objects:
            
            if target_obj == 'table':
                table_physx_view = self._physics_sim_view.create_rigid_body_view('/Replicator/Ref_Xform/Ref/material')
                poses = table_physx_view.get_transforms()[0]
                world_poses[target_obj] = {
                    'module_name': target_obj,
                    'world_position': [poses[0], poses[1], poses[2]],
                    'world_orientation': [poses[3], poses[4], poses[5], poses[6]] # xyzw
                }
            # elif target_obj == 'box':
            #     box_physx_view = self._physics_sim_view.create_rigid_body_view('/Replicator/Ref_Xform_01/Ref/Cardboard_Box___12x9x4')
            #     poses = box_physx_view.get_transforms()[0]
            #     world_poses[target_obj] = {
            #         'module_name': target_obj,
            #         'world_position': [poses[0], poses[1], poses[2]],
            #         'world_orientation': [poses[3], poses[4], poses[5], poses[6]]
            #     }
        return world_poses

    def get_target_object_transforms(self, step_size=None):
        """Get transforms of target objects and their children in world coordinate system."""
        poses_data = self.setup_target_objects_rigidbody()
        
        # Log to CSV if enabled
        self.pose_logger.log_poses(poses_data)
            
        return poses_data


    @staticmethod
    def _euler_to_quat_wxyz(euler_xyz_rad):
        """欧拉角 (XYZ, 弧度) → 四元数 [w, x, y, z]"""
        cx, cy, cz = np.cos(euler_xyz_rad / 2)
        sx, sy, sz = np.sin(euler_xyz_rad / 2)
        return np.array([
            cx*cy*cz + sx*sy*sz,
            sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz,
        ], dtype=np.float64)

    def _delete_old_parts(self):
        """删除旧零件：清除 rigid prim 缓存 → 从 USD stage 删除 prim"""
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # 先清除缓存，确保无 tensor view 引用旧 prim
        self._parts_rigid_prims = []
        self._rigid_body_paths = []
        self._rigid_prims_initialized = False

        for path in self.parts_prim_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
        print(f"[SceneBuilder] 已删除 {len(self.parts_prim_paths)} 个旧零件")

    def _create_parts_at_paths(self, part_pools, target_paths, plane_index=0):
        """在指定的 prim 路径上创建新零件并随机散布。

        Args:
            part_pools: list of (pool_or_usd, count) — 每组的资产池和数量
            target_paths: 目标 prim 路径列表，新零件将创建在这些路径上
            plane_index: 散布目标平面索引
        """
        import omni.usd
        from pxr import UsdPhysics, UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()

        center = np.array(self.plane_cfg['plane_position'][plane_index], dtype=np.float64)
        scale = np.array(self.plane_cfg['plane_scale'][plane_index], dtype=np.float64)
        half_x, half_y = scale[0] * 0.5, scale[1] * 0.5

        idx = 0
        for pool, count in part_pools:
            for _ in range(count):
                if idx >= len(target_paths):
                    print(f"[SceneBuilder] 警告: 目标路径不足，已创建 {idx} 个")
                    return
                prim_path = target_paths[idx]
                idx += 1

                if isinstance(pool, list):
                    usd_file = self._usd_path(random.choice(pool))
                else:
                    usd_file = self._usd_path(pool)

                stage_utils.add_reference_to_stage(usd_path=usd_file, prim_path=prim_path)
                prim = stage.GetPrimAtPath(prim_path)

                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(self.part_cfg.get('mass', 0.2))

                pos = Gf.Vec3d(
                    float(center[0] + random.uniform(-half_x, half_x)),
                    float(center[1] + random.uniform(-half_y, half_y)),
                    float(center[2] + 0.03),
                )
                euler = np.array([
                    random.uniform(-np.pi / 2, np.pi / 2),
                    random.uniform(-np.pi / 2, np.pi / 2),
                    random.uniform(-np.pi / 2, np.pi / 2),
                ])
                quat = self._euler_to_quat_wxyz(euler)

                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(pos)
                xformable.AddOrientOp().Set(
                    Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
                )

    def _randomize_task1_assets(self):
        """删除旧零件，从资产池中随机选择并在相同路径上创建新零件（仅 USD 层，位姿在 world.reset 后由 scatter_after_reset 设置）"""
        saved_paths = list(self._initial_parts_prim_paths)
        self._delete_old_parts()

        part_a_pool = self.part_cfg.get('part_a_assets', [])
        part_b_pool = self.part_cfg.get('part_b_assets', [])
        num_a = self.part_cfg.get('num_parts', 2)

        self._create_parts_at_paths(
            part_pools=[(part_a_pool, num_a), (part_b_pool, num_a)],
            target_paths=saved_paths,
            plane_index=0,
        )

        self.parts_prim_paths = saved_paths
        self._rigid_prims_initialized = False
        print(f"[SceneBuilder] Task1 已重新创建 {len(saved_paths)} 个零件 (路径不变，等待 scatter)")

    def _randomize_task3_assets(self):
        """删除旧零件，在相同路径上为每个箱子重新创建零件（仅 USD 层，位姿在 world.reset 后由 scatter_after_reset 设置）"""
        saved_paths = list(self._initial_parts_prim_paths)
        self._delete_old_parts()

        # 获取 part A / B 资产（支持 pool 或单个 usd）
        part_a = self.part_cfg.get('part_a_assets', self.part_cfg.get('part_a_usd', ''))
        part_b = self.part_cfg.get('part_b_assets', self.part_cfg.get('part_b_usd', ''))
        part_usds = [part_a, part_b]

        num_groups = len(self.box_cfg['box_position'])
        parts_per_group = 3

        for i in range(num_groups):
            asset = part_usds[i] if i < len(part_usds) else part_usds[-1]
            start = i * parts_per_group
            end = start + parts_per_group
            group_target_paths = saved_paths[start:end]
            self._create_parts_at_paths(
                part_pools=[(asset, len(group_target_paths))],
                target_paths=group_target_paths,
                plane_index=i,
            )

        self.parts_prim_paths = saved_paths
        self._rigid_prims_initialized = False
        print(f"[SceneBuilder] Task3 已重新创建 {len(saved_paths)} 个零件到 {num_groups} 个平面 (路径不变，等待 scatter)")

    def scatter_after_reset(self):
        """在 world.reset() 之后调用：用 SingleRigidPrim 设置随机位姿 + 清速度。

        此时物理视图已重建，SingleRigidPrim 可正常工作。
        """
        task = self.cfg['task_number']

        # 重建 rigid prim 缓存（world.reset 后 tensor view 已失效）
        self._rebuild_rigid_prims(self.parts_prim_paths)

        if task == 1:
            self._scatter_parts_direct(plane_index=0)
            print(f"[SceneBuilder] Task1 scatter_after_reset: 已随机散布 {len(self.parts_prim_paths)} 个零件")
        elif task == 2:
            # world.reset() 后物理视图已就绪，用随机顺序设置初始位姿并清零速度
            total_parts = len(self.parts_prim_paths)
            random_indices = np.random.permutation(np.arange(total_parts))
            self.rigid_prim.set_world_poses(
                positions=self._task2_initial_positions,
                indices=random_indices
            )
            self.rigid_prim.set_velocities(velocities=np.zeros((total_parts, 6)))
            # 更新本轮随机顺序，供 reset() 复用
            self._task2_initial_indices = random_indices.copy()
            print(f"[SceneBuilder] Task2 scatter_after_reset: 已设置 {total_parts} 个零件初始位姿")
        elif task == 3:
            num_groups = len(self.box_cfg['box_position'])
            parts_per_group = 3
            for i in range(num_groups):
                start = i * parts_per_group
                end = start + parts_per_group
                group_paths = self.parts_prim_paths[start:end]
                self._scatter_parts_direct(plane_index=i, prim_paths=group_paths)
            print(f"[SceneBuilder] Task3 scatter_after_reset: 已随机散布 {len(self.parts_prim_paths)} 个零件到 {num_groups} 个平面")

    def _scatter_parts_direct(self, plane_index=0, prim_paths=None):
        """直接通过 SingleRigidPrim 随机散布零件（完全绕过 Replicator）。

        散布范围与初始化时 scatter_2d 一致：
        - Isaac Sim Z-up，平面在 XY 平面上
        - 平面基底 1×1，缩放后 X 方向范围 = scale[0]，Y 方向范围 = scale[1]
        - Z = 平面高度 + 小偏移（防止穿模）
        """
        if prim_paths is None:
            prim_paths = self.parts_prim_paths

        center = np.array(self.plane_cfg['plane_position'][plane_index], dtype=np.float64)
        scale = np.array(self.plane_cfg['plane_scale'][plane_index], dtype=np.float64)
        # 散布范围与 scatter_2d 一致：平面 X/Y 方向各 scale*0.5
        half_x = scale[0] * 0.5
        half_y = scale[1] * 0.5

        # 延迟初始化 + 使用缓存的 rigid prim
        self._ensure_rigid_prims()
        rigid_prims = getattr(self, '_parts_rigid_prims', [])
        all_paths = self.parts_prim_paths

        for path in prim_paths:
            # 从缓存中查找对应的 rigid prim
            try:
                idx = all_paths.index(path)
                rigid = rigid_prims[idx] if idx < len(rigid_prims) else None
            except (ValueError, IndexError):
                rigid = None
            if rigid is None:
                print(f"[SceneBuilder] _scatter_parts_direct: 无缓存刚体 {path}")
                continue

            pos = np.array([
                center[0] + random.uniform(-half_x, half_x),
                center[1] + random.uniform(-half_y, half_y),
                center[2] + 0.03,   # Z 方向稍高于平面，避免穿模
            ], dtype=np.float64)
            euler = np.array([
                random.uniform(-np.pi/2, np.pi/2),
                random.uniform(-np.pi/2, np.pi/2),
                random.uniform(-np.pi/2, np.pi/2),
            ])
            quat = self._euler_to_quat_wxyz(euler)

            rigid.set_world_pose(position=pos, orientation=quat)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))

        print(f"[SceneBuilder] _scatter_parts_direct: 已随机散布 {len(prim_paths)} 个零件到平面 {plane_index}")

    def _clear_parts_velocities(self):
        """清零所有零件的线速度和角速度（使用缓存的 rigid prim）"""
        self._ensure_rigid_prims()
        rigid_prims = getattr(self, '_parts_rigid_prims', [])
        for i, part_path in enumerate(self.parts_prim_paths):
            if i < len(rigid_prims) and rigid_prims[i] is not None:
                try:
                    rigid_prims[i].set_linear_velocity(np.zeros(3))
                    rigid_prims[i].set_angular_velocity(np.zeros(3))
                except Exception as e:
                    print(f"[SceneBuilder] 清零 {part_path} 速度失败: {e}")
            else:
                print(f"[SceneBuilder] 清零 {part_path} 速度跳过: 无缓存 rigid body")

    def _reset_boxes(self):
        """重置 Task 1/2/3 的箱子到配置中的初始位置"""
        if hasattr(self, 'boxes') and hasattr(self, 'box_cfg'):
            self.boxes.set_world_poses(
                positions=np.array(self.box_cfg['box_position']),
            )
            print(f"[SceneBuilder] 箱子已重置到初始位置")


    def save_parts_poses(self, save_dir: Optional[Path] = None) -> Optional[Path]:
        """
        保存所有零件的世界位姿数据到 JSON 文件（适用于所有任务）

        功能说明：
            1. 调用 get_parts_world_poses 获取当前所有零件的实时世界位姿
            2. 自动生成带时间戳和任务编号的文件名，避免覆盖
            3. 将位姿数据格式化后保存为 JSON 文件
            4. 完善异常处理，保证保存失败时不崩溃并输出错误日志
            5. 返回最终保存的文件路径，方便外部调用

        参数说明：
            save_dir (Optional[Path]): 可选参数，指定保存目录
                - 不传入时：默认保存到项目根目录下的 outputs 子目录
                - 传入时：使用指定目录（自动创建不存在的目录）

        返回值：
            Optional[Path]: 成功返回保存的文件完整路径，失败返回 None
        """
        try:
            # 1. 获取零件位姿数据
            all_parts_poses: List[Dict] = self.get_parts_world_poses()

            # 空数据判断
            if not all_parts_poses:
                print("[save_parts_poses] 错误：未获取到任何零件位姿数据，取消保存")
                return None

            # 2. 生成保存路径
            # 根据当前任务编号和时间戳生成文件名
            task_num = self.cfg.get('task_number', 1)
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"task{task_num}_parts_poses_{time_stamp}.json"

            # 确定保存目录
            if save_dir is None:
                # 默认路径：项目根目录下的 outputs 子目录
                save_dir = Path(__file__).parent.parent.parent / "outputs"
            else:
                # 确保传入的路径是 Path 类型
                save_dir = Path(save_dir)

            # 自动创建目录
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / file_name

            # 3. 写入 JSON 文件（格式化输出，方便阅读）
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(
                    all_parts_poses,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            # 4. 保存成功日志
            print(f"[save_parts_poses] 保存成功！共 {len(all_parts_poses)} 个零件")
            print(f"[save_parts_poses] 文件路径：{save_path.absolute()}")
            return save_path

        except Exception as e:
            # 全局异常捕获，保证函数健壮性
            print(f"[save_parts_poses] 保存失败：{str(e)}")
            return None

            


    def reset(self):
        """重置场景状态，所有任务的物体恢复到初始随机化/固定位置"""
        task = self.cfg['task_number']
        print(f"[SceneBuilder] 重置任务{task}")

        # ── 任务1重置：直接通过 SingleRigidPrim 散布（完全绕过 Replicator） ──
        if task == 1:
            self.save_parts_poses()
            self._randomize_task1_assets()
            self._reset_boxes()
            print("[SceneBuilder] Task1 已重置")
            return

        # ── 任务2重置：传送带零件恢复初始位置 + 重新随机排列顺序 ──
        if task == 2:
            # 传送带表面速度：Isaac Sim 5.1 / PhysX 107 使用 PhysxSurfaceVelocityAPI（Vector3f）。
            def _set_conveyor_surface_velocity(velocity_vec3) -> None:
                import omni.usd
                from pxr import PhysxSchema
                stage = omni.usd.get_context().get_stage()
                for prim in stage.Traverse():
                    if "ConveyorBelt" not in str(prim.GetPath()):
                        continue
                    if prim.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI):
                        PhysxSchema.PhysxSurfaceVelocityAPI(prim).GetSurfaceVelocityAttr().Set(
                            velocity_vec3
                        )

            try:
                from pxr import Gf
                _set_conveyor_surface_velocity(Gf.Vec3f(0.0, 0.0, 0.0))
            except Exception as e:
                print(f"[SceneBuilder] 传送带停止失败: {e}")

            total_parts = len(self.parts_prim_paths)
            random_indices = np.random.permutation(np.arange(total_parts))
            self.rigid_prim.set_velocities(velocities=np.zeros((total_parts, 6)))
            self.rigid_prim.set_world_poses(
                positions=self._task2_initial_positions,
                indices=random_indices
            )
            self.rigid_prim.set_velocities(velocities=np.zeros((total_parts, 6)))

            # 重新启动传送带（沿 X 轴正方向，速度 0.1 m/s）
            try:
                from pxr import Gf
                _set_conveyor_surface_velocity(Gf.Vec3f(0.1, 0.0, 0.0))
            except Exception as e:
                print(f"[SceneBuilder] 传送带重启失败: {e}")

            self.save_parts_poses()
            self._reset_boxes()
            print("[SceneBuilder] Task2 已重置")
            return

        # ── 任务3重置：删除旧零件，重新创建并随机散布 ──
        if task == 3:
            self._randomize_task3_assets()
            self._reset_boxes()
            if self.box_cfg.get('lock_boxes', False):
                self._lock_box_positions()
            print("[SceneBuilder] Task3 已重置")
            return

        # ── 任务4重置：重置箱子位置 ──
        if task == 4:
            if hasattr(self, 'box_articulation') and self.box_articulation is not None:
                # 重置关节位置（仅当为 Articulation 且有关节时）
                if (self.box_initial_joint_positions is not None
                        and hasattr(self.box_articulation, 'set_joint_positions')):
                    self.box_articulation.set_joint_positions(self.box_initial_joint_positions)
                # 使用世界坐标重置箱子位置
                self.box_articulation.set_world_poses(
                    positions=self._box_initial_world_pos,
                    orientations=self._box_initial_world_ori
                )
                # 清零箱子速度（仅当物理对象支持时）
                if hasattr(self.box_articulation, 'set_velocities'):
                    self.box_articulation.set_velocities(velocities=np.zeros((1, 6)))
            print("[SceneBuilder] Task4 已重置")
            return

        print(f"[SceneBuilder] 警告: 未知的 task_number={task}，跳过重置")
