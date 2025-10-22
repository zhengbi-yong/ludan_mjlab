"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command
