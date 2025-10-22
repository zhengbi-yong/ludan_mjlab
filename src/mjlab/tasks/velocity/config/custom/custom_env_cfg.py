"""Velocity-tracking environment configuration for user-provided robots."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.asset_zoo.robots.custom_robot import (
  CustomRobotAssetCfg,
  make_custom_robot_entity_cfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg


@dataclass
class CustomRobotVelocityEnvCfg(LocomotionVelocityEnvCfg):
  """Environment configuration that defers robot details to runtime."""

  robot: CustomRobotAssetCfg = field(default_factory=CustomRobotAssetCfg)

  def __post_init__(self) -> None:  # noqa: D401 - inherited docstring is sufficient.
    super().__post_init__()

    entity_cfg, contact_sensor_names = make_custom_robot_entity_cfg(self.robot)

    scene_entities = dict(self.scene.entities)
    scene_entities["robot"] = entity_cfg
    self.scene.entities = scene_entities

    if self.robot.action_scale is not None:
      self.actions.joint_pos.scale = self.robot.action_scale

    if self.robot.viewer_body_name:
      self.viewer.body_name = self.robot.viewer_body_name

    if self.robot.command_viz_height is not None:
      self.commands.twist.viz.z_offset = self.robot.command_viz_height

    if self.robot.pose_stds is not None:
      self.rewards.pose.params["std"] = self.robot.pose_stds

    if contact_sensor_names:
      self.rewards.air_time.params["sensor_names"] = list(contact_sensor_names)

    if self.events.foot_friction is not None:
      asset_cfg = self.events.foot_friction.params.get("asset_cfg")
      if asset_cfg is not None:
        asset_cfg.geom_names = list(self.robot.foot_geom_names)

    if self.robot.disable_push_events:
      self.events.push_robot = None


@dataclass
class CustomRobotVelocityEnvCfg_PLAY(CustomRobotVelocityEnvCfg):
  """Play configuration with effectively unbounded episode length."""

  def __post_init__(self) -> None:  # noqa: D401
    super().__post_init__()
    self.episode_length_s = int(1e9)
