"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class RslRlPpoActorCriticCfg:
  """Config for the PPO actor-critic networks."""

  init_noise_std: float = 1.0
  """The initial noise standard deviation of the policy."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """The type of noise standard deviation for the policy. Default is scalar."""
  actor_obs_normalization: bool = False
  """Whether to normalize the observation for the actor network. Default is False."""
  critic_obs_normalization: bool = False
  """Whether to normalize the observation for the critic network. Default is False."""
  actor_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the actor network."""
  critic_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the critic network."""
  activation: str = "elu"
  """The activation function to use in the actor and critic networks."""
  class_name: str = "ActorCritic"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  class_name: str = "PPO"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, list[str]] = field(
    default_factory=lambda: {"policy": ["policy"], "critic": ["policy", "critic"]},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """The experiment name."""
  run_name: str = ""
  """The run name. Default is empty string."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  policy: RslRlPpoActorCriticCfg = field(default_factory=RslRlPpoActorCriticCfg)
  """The policy configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""
