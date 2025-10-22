import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.envs import types
from mjlab.envs.manager_based_env import ManagerBasedEnv, ManagerBasedEnvCfg
from mjlab.managers.command_manager import CommandManager, NullCommandManager
from mjlab.managers.curriculum_manager import CurriculumManager, NullCurriculumManager
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.logging import print_info
from mjlab.viewer.offscreen_renderer import OffscreenRenderer


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
  """Configuration for a manager-based RL environment."""

  episode_length_s: float
  rewards: Any
  terminations: Any
  commands: Any | None = None
  curriculum: Any | None = None
  is_finite_horizon: bool = False


class ManagerBasedRlEnv(ManagerBasedEnv, gym.Env):
  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }

  cfg: ManagerBasedRlEnvCfg

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    super().__init__(cfg=cfg, device=device)
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore

    print_info("[INFO]: Completed setting up the environment...")

  # Properties.

  @property
  def max_episode_length_s(self) -> float:
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> int:
    return math.ceil(self.max_episode_length_s / self.step_dt)

  # Methods.

  def setup_manager_visualizers(self) -> None:
    self.manager_visualizers = {}
    if getattr(self.command_manager, "active_terms", None):
      self.manager_visualizers["command_manager"] = self.command_manager

  def load_managers(self) -> None:
    # NOTE: Order is important.
    if self.cfg.commands is not None:
      self.command_manager = CommandManager(self.cfg.commands, self)
    else:
      self.command_manager = NullCommandManager()
    print_info(f"[INFO] Command Manager: {self.command_manager}")
    super().load_managers()
    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    print_info(f"[INFO] Termination Manager: {self.termination_manager}")
    self.reward_manager = RewardManager(self.cfg.rewards, self)
    print_info(f"[INFO] Reward Manager: {self.reward_manager}")
    if self.cfg.curriculum is not None:
      self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
    else:
      self.curriculum_manager = NullCurriculumManager()
    print_info(f"[INFO] Curriculum Manager: {self.curriculum_manager}")
    self._configure_gym_env_spaces()
    if "startup" in self.event_manager.available_modes:
      self.event_manager.apply(mode="startup")
      self.sim.create_graph()

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:  # pyright: ignore[reportIncompatibleMethodOverride]
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.obs_buf = self.observation_manager.compute(update_history=True)

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def render(self) -> np.ndarray | None:
    if self.render_mode == "human" or self.render_mode is None:
      return None
    elif self.render_mode == "rgb_array":
      if self._offline_renderer is None:
        raise ValueError("Offline renderer not initialized")
      debug_callback = (
        self.update_visualizers if hasattr(self, "update_visualizers") else None
      )
      self._offline_renderer.update(self.sim.data, debug_vis_callback=debug_callback)
      return self._offline_renderer.render()
    else:
      raise NotImplementedError(
        f"Render mode {self.render_mode} is not supported. "
        f"Please use: {self.metadata['render_modes']}."
      )

  def close(self) -> None:
    if self._offline_renderer is not None:
      self._offline_renderer.close()
    super().close()

  # Private methods.

  def _configure_gym_env_spaces(self) -> None:
    self.single_observation_space = gym.spaces.Dict()
    for group_name, group_term_names in self.observation_manager.active_terms.items():
      has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
      group_dim = self.observation_manager.group_obs_dim[group_name]
      if has_concatenated_obs:
        assert isinstance(group_dim, tuple)
        self.single_observation_space[group_name] = gym.spaces.Box(
          low=-math.inf, high=math.inf, shape=group_dim
        )
      else:
        assert not isinstance(group_dim, tuple)
        group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
        for term_name, term_dim, _term_cfg in zip(
          group_term_names, group_dim, group_term_cfgs, strict=False
        ):
          self.single_observation_space[group_name] = gym.spaces.Dict(
            {term_name: gym.spaces.Box(low=-math.inf, high=math.inf, shape=term_dim)}
          )

    action_dim = sum(self.action_manager.action_term_dim)
    self.single_action_space = gym.spaces.Box(
      low=-math.inf, high=math.inf, shape=(action_dim,)
    )

    self.observation_space = gym.vector.utils.batch_space(
      self.single_observation_space, self.num_envs
    )
    self.action_space = gym.vector.utils.batch_space(
      self.single_action_space, self.num_envs
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.curriculum_manager.compute(env_ids=env_ids)
    # Reset the internal buffers of the scene elements.
    self.scene.reset(env_ids)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # NOTE: This is order sensitive.
    self.extras["log"] = dict()
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
