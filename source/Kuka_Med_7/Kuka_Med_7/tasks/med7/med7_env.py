
from __future__ import annotations

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

        # Static props (visual + collision, spawned in env template)
        self.cfg.robot_mount.spawn.func(
            self.cfg.robot_mount.prim_path,
            self.cfg.robot_mount.spawn,
            translation=self.cfg.robot_mount.init_state.pos,
            orientation=self.cfg.robot_mount.init_state.rot,
        )
        self.cfg.hospital_bed.spawn.func(
            self.cfg.hospital_bed.prim_path,
            self.cfg.hospital_bed.spawn,
            translation=self.cfg.hospital_bed.init_state.pos,
            orientation=self.cfg.hospital_bed.init_state.rot,
        )

        # Lights
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)
        self.cfg.distant_light.spawn.func(
            self.cfg.distant_light.prim_path,
            self.cfg.distant_light.spawn,
            orientation=self.cfg.distant_light.init_state.rot,
        )

        # Ground plane
        self.cfg.ground_plane.spawn.func(
            self.cfg.ground_plane.prim_path, self.cfg.ground_plane.spawn
        )

        # Hand Visualization Proxy (RigidObject)
        from isaaclab.assets import RigidObject
        self.hand_proxy = RigidObject(self.cfg.hand_proxy)

        # Cameras
        from isaaclab.sensors import Camera
        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.room_camera = Camera(self.cfg.room_camera)

        # Clone environments and register assets
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["hand_proxy"] = self.hand_proxy
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["room_camera"] = self.room_camera

        self.scene.filter_collisions(global_prim_paths=[])

        # ---------------------------------------------------------------
        # Spawn bones OUTSIDE /World/envs/ so PhysX never touches them.
        # They live at /World/Bones/femur and /World/Bones/tibia.
        # ---------------------------------------------------------------
        import isaaclab.sim as sim_utils
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()

        # Create a top-level scope that is NOT under the physics scene
        bones_scope_path = "/World/Bones"
        if not stage.GetPrimAtPath(bones_scope_path).IsValid():
            UsdGeom.Scope.Define(stage, bones_scope_path)

        femur_init_pos = self.cfg.femur.init_state.pos   # (x, y, z) tuple
        femur_init_rot = self.cfg.femur.init_state.rot   # (w, x, y, z) tuple
        tibia_init_pos = self.cfg.tibia.init_state.pos
        tibia_init_rot = self.cfg.tibia.init_state.rot

        self.cfg.femur.spawn.func(
            "/World/Bones/femur",
            self.cfg.femur.spawn,
            translation=femur_init_pos,
            orientation=femur_init_rot,
        )
        self.cfg.tibia.spawn.func(
            "/World/Bones/tibia",
            self.cfg.tibia.spawn,
            translation=tibia_init_pos,
            orientation=tibia_init_rot,
        )

        # Store the prim paths for runtime updates
        self._femur_prim_path = "/World/Bones/femur"
        self._tibia_prim_path = "/World/Bones/tibia"

        print(f"[INFO] Femur spawned at: {self._femur_prim_path}")
        print(f"[INFO] Tibia  spawned at: {self._tibia_prim_path}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

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

    def update_femur_pose(self, pos: torch.Tensor, quat: torch.Tensor):
        """Move the femur visual mesh to the given world pose."""
        self._set_xform_pose(self._femur_prim_path, pos, quat)

    def update_tibia_pose(self, pos: torch.Tensor, quat: torch.Tensor):
        """Move the tibia visual mesh to the given world pose."""
        self._set_xform_pose(self._tibia_prim_path, pos, quat)
