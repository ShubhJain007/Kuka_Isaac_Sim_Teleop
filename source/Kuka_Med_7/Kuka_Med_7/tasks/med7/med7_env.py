
from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .med7_env_cfg import Med7EnvCfg


class Med7Env(DirectRLEnv):
    cfg: Med7EnvCfg

    def __init__(self, cfg: Med7EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup the scene."""
        # Robot (articulation)
        self.robot = Articulation(self.cfg.robot)

        # Lights
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)
        self.cfg.distant_light.spawn.func(
            self.cfg.distant_light.prim_path,
            self.cfg.distant_light.spawn,
            orientation=self.cfg.distant_light.init_state.rot,
        )

        # Ground plane — must pass translation so it sits below the bed
        self.cfg.ground_plane.spawn.func(
            self.cfg.ground_plane.prim_path, self.cfg.ground_plane.spawn,
            translation=self.cfg.ground_plane.init_state.pos,
        )

        # Black cuboid pedestal (arm mount). Scene cfg fields aren't auto-spawned;
        # they are only materialized when spawn.func is invoked explicitly here.
        self._base_pedestal_prim_path = "/World/envs/env_0/BasePedestal"
        if hasattr(self.cfg, "base_pedestal") and self.cfg.base_pedestal is not None:
            self.cfg.base_pedestal.spawn.func(
                self.cfg.base_pedestal.prim_path, self.cfg.base_pedestal.spawn,
                translation=self.cfg.base_pedestal.init_state.pos,
            )
            print(f"[INFO] Spawned base pedestal: {self._base_pedestal_prim_path}")

        # Hospital bed removed — replaced by the real physical table in AR mode.
        if hasattr(self.cfg, "hospital_bed") and self.cfg.hospital_bed is not None:
            self.cfg.hospital_bed.spawn.func(
                self.cfg.hospital_bed.prim_path, self.cfg.hospital_bed.spawn,
                translation=self.cfg.hospital_bed.init_state.pos,
            )

        # Hand Visualization Proxy (RigidObject)
        from isaaclab.assets import RigidObject
        self.hand_proxy = RigidObject(self.cfg.hand_proxy)

        # Surgical probe (RigidObject — registered so avp_stream tracks its pose)
        self.probe_obj = RigidObject(self.cfg.probe)

        # End-effector mount (RigidObject, kinematic — avp_stream streams it per frame)
        self.ee_mount = RigidObject(self.cfg.ee_mount)

        # Clone environments and register assets
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["hand_proxy"] = self.hand_proxy
        self.scene.rigid_objects["probe"] = self.probe_obj
        self.scene.rigid_objects["ee_mount"] = self.ee_mount

        self.scene.filter_collisions(global_prim_paths=[])

        # Spawn bone USD meshes — visible from start, included in USDZ for AVP.
        # When ROS point cloud data arrives, the teleop script moves these prims
        # via ICP to match the tracked bone position.
        self._femur_prim_path = "/World/envs/env_0/femur"
        self._tibia_prim_path = "/World/envs/env_0/tibia"

        # DUMMY SPACER TEST (2026-04-22): empty Xform between EEMount and femur.
        # Tests whether USD-graph adjacency to EEMount (whose transform is rewritten
        # every frame) is what causes the femur jitter.
        # REVERT: delete this import + Define block.
        import omni.usd as _omni_usd_dbg
        from pxr import UsdGeom as _UsdGeom_dbg
        _stage_dbg = _omni_usd_dbg.get_context().get_stage()
        _UsdGeom_dbg.Xform.Define(_stage_dbg, "/World/envs/env_0/ZZZSpacer")
        print("[DBG] Spawned spacer prim /World/envs/env_0/ZZZSpacer (between EEMount and femur)")

        if hasattr(self.cfg, "femur") and self.cfg.femur is not None:
            self.cfg.femur.spawn.func(
                self.cfg.femur.prim_path,
                self.cfg.femur.spawn,
                translation=self.cfg.femur.init_state.pos,
                orientation=self.cfg.femur.init_state.rot,
            )
            print(f"[INFO] Spawned femur USD: {self._femur_prim_path}")

        if hasattr(self.cfg, "tibia") and self.cfg.tibia is not None:
            self.cfg.tibia.spawn.func(
                self.cfg.tibia.prim_path,
                self.cfg.tibia.spawn,
                translation=self.cfg.tibia.init_state.pos,
                orientation=self.cfg.tibia.init_state.rot,
            )
            print(f"[INFO] Spawned tibia USD: {self._tibia_prim_path}")

    # Offset from lbr_link_7 frame to the mount visual origin.
    # xyz=(0, 0, 0.189), rpy=(0, π, 0)  →  quat (w,x,y,z) = (0, 0, 1, 0)
    _EE_OFFSET_POS = (0.0, 0.0, 0.189)
    _EE_OFFSET_QUAT = (0.0, 0.0, 1.0, 0.0)

    def _sync_ee_mount_pose(self) -> None:
        """Compose lbr_link_7 world pose with the EE offset and write to the EEMount rigid body."""
        if not hasattr(self, "ee_mount"):
            return
        try:
            link7_idx = self.robot.data.body_names.index("lbr_link_7")
        except ValueError:
            return

        body_pose = self.robot.data.body_state_w[0, link7_idx, :7].detach().cpu().numpy()
        p7 = body_pose[:3]
        w, x, y, z = float(body_pose[3]), float(body_pose[4]), float(body_pose[5]), float(body_pose[6])

        ox, oy, oz = self._EE_OFFSET_POS
        ow, oxq, oyq, ozq = self._EE_OFFSET_QUAT

        # Rotate offset translation by q7
        tx = (1 - 2 * (y * y + z * z)) * ox + 2 * (x * y - z * w) * oy + 2 * (x * z + y * w) * oz
        ty = 2 * (x * y + z * w) * ox + (1 - 2 * (x * x + z * z)) * oy + 2 * (y * z - x * w) * oz
        tz = 2 * (x * z - y * w) * ox + 2 * (y * z + x * w) * oy + (1 - 2 * (x * x + y * y)) * oz

        # Hamilton product q7 * q_offset
        qw = w * ow - x * oxq - y * oyq - z * ozq
        qx = w * oxq + x * ow + y * ozq - z * oyq
        qy = w * oyq - x * ozq + y * ow + z * oxq
        qz = w * ozq + x * oyq - y * oxq + z * ow

        pose = torch.tensor(
            [[float(p7[0]) + tx, float(p7[1]) + ty, float(p7[2]) + tz, qw, qx, qy, qz]],
            dtype=torch.float32,
            device=self.device,
        )
        self.ee_mount.write_root_pose_to_sim(pose)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._sync_ee_mount_pose()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        return {"policy": torch.cat((self.joint_pos, self.joint_vel), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Bone pose update helpers (pure USD transform — no physics API)
    # ------------------------------------------------------------------

    def _set_xform_pose(self, prim_path: str, pos: torch.Tensor, quat: torch.Tensor):
        """Set a USD Xform prim's pose directly via matrix transform.

        Because the bone prims live outside /World/envs/, the physics solver
        never touches them and there is no jitter.

        Args:
            prim_path: Full USD prim path, e.g. '/World/Bones/femur'.
            pos:  Position tensor (1, 3) — (x, y, z) in world frame.
            quat: Quaternion tensor (1, 4) — (w, x, y, z) in world frame.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        pos_np = pos[0].cpu().numpy()
        q     = quat[0].cpu().numpy()   # (w, x, y, z)

        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Prim not found: {prim_path}")
            return

        rotation    = Gf.Rotation(Gf.Quaternion(float(q[0]),
                                                 Gf.Vec3d(float(q[1]),
                                                          float(q[2]),
                                                          float(q[3]))))
        translation = Gf.Vec3d(float(pos_np[0]), float(pos_np[1]), float(pos_np[2]))

        matrix = Gf.Matrix4d()
        matrix.SetRotateOnly(rotation)
        matrix.SetTranslateOnly(translation)

        UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)

    def set_bone_xform(self, bone: str, matrix_4x4: "np.ndarray"):
        """Move a bone USD prim using a 4x4 world-frame transform.

        Args:
            bone: "femur" or "tibia"
            matrix_4x4: 4x4 numpy array (standard math convention, column 3 = translation)
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        prim_path = self._femur_prim_path if bone == "femur" else self._tibia_prim_path
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        T = matrix_4x4
        gf_mat = Gf.Matrix4d(
            T[0, 0], T[1, 0], T[2, 0], T[3, 0],
            T[0, 1], T[1, 1], T[2, 1], T[3, 1],
            T[0, 2], T[1, 2], T[2, 2], T[3, 2],
            T[0, 3], T[1, 3], T[2, 3], T[3, 3],
        )
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(gf_mat)
    
    def sync_robot_joint_state(self, joint_names: list[str], positions: torch.Tensor):
        """Sync robot joint positions from ROS to Isaac Sim.
        
        Args:
            joint_names: List of joint names from ROS
            positions: Tensor of joint positions (shape: [num_joints])
        """
        # Map ROS joint names to Isaac Sim joint indices
        robot_joint_names = self.robot.data.joint_names
        
        # Clone to avoid in-place modification of inference tensors
        joint_pos = self.robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(self.robot.data.joint_vel)  # Zero velocity for syncing
        
        for i, ros_name in enumerate(joint_names):
            # Try to find matching joint in Isaac Sim
            for j, sim_name in enumerate(robot_joint_names):
                if ros_name in sim_name or sim_name in ros_name:
                    if i < len(positions):
                        joint_pos[0, j] = positions[i]
                    break
        
        # Write to simulation (outside inference mode, uses cloned tensors)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
