
import gymnasium as gym

from .med7_env import Med7Env, Med7EnvCfg

gym.register(
    id="Isaac-Med7-v0",
    entry_point="Kuka_Med_7.tasks.med7.med7_env:Med7Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Med7EnvCfg,
    },
)
