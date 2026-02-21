
"""
Script to run the Kuka Med 7 environment.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run the Kuka Med 7 environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import Kuka_Med_7  # noqa: F401
from Kuka_Med_7.tasks.med7.med7_env_cfg import Med7EnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = Med7EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # create environment
    env = gym.make("Isaac-Med7-v0", cfg=env_cfg)

    # simulate physics
    obs, _ = env.reset()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from policy
            # For now, just sample random actions
            actions = torch.rand(env.unwrapped.num_envs, env.unwrapped.action_space.shape[0], device=env.unwrapped.device)
            actions = actions * 2 - 1 # scale to [-1, 1]
            
            # step
            obs, rew, terminated, truncated, info = env.step(actions)
            
    # close the environment
    env.close()


if __name__ == "__main__":
    main()
