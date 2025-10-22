"""Gym registrations for custom robot velocity tracking tasks."""

from __future__ import annotations

import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Flat-Custom", 
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.custom_env_cfg:CustomRobotVelocityEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:CustomRobotPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Custom-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.custom_env_cfg:CustomRobotVelocityEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:CustomRobotPPORunnerCfg",
  },
)
