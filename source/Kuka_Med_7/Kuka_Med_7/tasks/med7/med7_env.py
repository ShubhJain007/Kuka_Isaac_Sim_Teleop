
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
        """Setup the scene: robot, cuboid mount, cuboid bed, lights, ground."""
        # Robot (articulation)
        self.robot = Articulation(self.cfg.robot)

        # Spawn cuboid props via their spawn functions (they are static visuals/colliders)
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

        # Hand Visualization Proxy (RigidObject handles its own spawning)
        from isaaclab.assets import RigidObject
        self.hand_proxy = RigidObject(self.cfg.hand_proxy)

        # Cameras
        from isaaclab.sensors import Camera
        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.room_camera = Camera(self.cfg.room_camera)

        # Clone environments and register robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["hand_proxy"] = self.hand_proxy

        # Add cameras to scene
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["room_camera"] = self.room_camera

        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions which are joint position targets."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        """Get observations."""
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        observations = {
            "policy": torch.cat((self.joint_pos, self.joint_vel), dim=-1),
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done state."""
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments based on specified indices."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
