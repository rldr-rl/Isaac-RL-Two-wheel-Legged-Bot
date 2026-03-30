# Minimal step9-only registration. No other step files are required.
import gymnasium as gym

from . import agents
from .tumble_env_cfg import *

gym.register(
    id="Flamingo-LegliftWalk-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Leglift_walk_env_cfg:UnitreeGo2LegliftWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Leglift_walk_rsl_rl_ppo_cfg:UnitreeGo2LegliftWalkPPORunnerCfg",
    },
)

gym.register(
    id="Flamingo-LegliftWalk-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Leglift_walk_env_cfg:UnitreeGo2LegliftWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Leglift_walk_rsl_rl_ppo_cfg:UnitreeGo2LegliftWalkPPORunnerCfg",
    },
)

gym.register(
    id="Flamingo-Tumble-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "lab.flamingo.tasks.manager_based.locomotion.velocity.config.go2.tumble_env_cfg:UnitreeGo2TumbleEnvCfg",
        "rsl_rl_cfg_entry_point": "lab.flamingo.tasks.manager_based.locomotion.velocity.config.go2.agents.tumble_rsl_rl_ppo_cfg:UnitreeGo2TumblePPORunnerCfg",
    },
)