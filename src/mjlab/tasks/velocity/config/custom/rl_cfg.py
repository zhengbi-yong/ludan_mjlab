"""Default PPO configuration for custom robot velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class CustomRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  """Lightweight PPO defaults suitable for quick iteration."""

  policy: RslRlPpoActorCriticCfg = field(
    default_factory=lambda: RslRlPpoActorCriticCfg(
      init_noise_std=0.8,
      actor_obs_normalization=False,
      critic_obs_normalization=False,
      actor_hidden_dims=(256, 128, 64),
      critic_hidden_dims=(256, 128, 64),
      activation="elu",
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )
  experiment_name: str = "custom_robot_velocity"
  save_interval: int = 50
  num_steps_per_env: int = 24
  max_iterations: int = 10_000
