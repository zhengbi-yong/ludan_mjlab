from __future__ import annotations

from pathlib import Path

import pytest
import sys
import types

if "warp" not in sys.modules:
  sys.modules["warp"] = types.SimpleNamespace(
    config=types.SimpleNamespace(enable_backward=False, quiet=False)
  )

mujoco = pytest.importorskip("mujoco")


from mjlab.asset_zoo.robots.custom_robot import (
  CustomRobotAssetCfg,
  make_custom_robot_entity_cfg,
)
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.custom.custom_env_cfg import (
  CustomRobotVelocityEnvCfg,
  CustomRobotVelocityEnvCfg_PLAY,
)


@pytest.fixture()
def minimal_mjcf(tmp_path: Path) -> Path:
  xml = tmp_path / "robot.xml"
  xml.write_text(
    """
<mujoco model="two_link">
  <compiler angle="degree" coordinate="global"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="torso">
      <freejoint/>
      <geom name="torso_geom" type="capsule" fromto="0 0 0 0 0 0.4" size="0.05"/>
      <body name="thigh">
        <joint name="hip" type="hinge" limited="true" range="-45 45" axis="0 1 0"/>
        <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 0.4" size="0.03"/>
        <body name="foot">
          <joint name="knee" type="hinge" limited="true" range="-90 20" axis="0 1 0"/>
          <geom name="foot_geom" type="box" size="0.06 0.02 0.15"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
    """.strip()
  )
  return xml


def test_make_custom_robot_entity_cfg_generates_contact_sensors(minimal_mjcf: Path):
  robot_cfg = CustomRobotAssetCfg(
    mjcf_path=str(minimal_mjcf),
    contact_bodies=("foot",),
    foot_geom_names=("foot_geom",),
    viewer_body_name="torso",
    effort_limit=10.0,
    stiffness=5.0,
    damping=0.5,
  )

  entity_cfg, sensor_names = make_custom_robot_entity_cfg(robot_cfg)

  assert sensor_names == ("foot_contact",)
  assert entity_cfg.sensors and entity_cfg.sensors[0].name == "foot_contact"
  assert entity_cfg.articulation is not None
  assert entity_cfg.articulation.actuators[0].stiffness == pytest.approx(5.0)

  entity = Entity(entity_cfg)
  assert entity.joint_names == ["hip", "knee"]


def test_custom_velocity_env_cfg_populates_scene(minimal_mjcf: Path, monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setenv("MJLAB_CUSTOM_ROBOT_XML", str(minimal_mjcf))

  env_cfg = CustomRobotVelocityEnvCfg(
    robot=CustomRobotAssetCfg(
      contact_bodies=("foot",),
      foot_geom_names=("foot_geom",),
      viewer_body_name="torso",
      command_viz_height=0.5,
      disable_push_events=True,
    )
  )

  assert "robot" in env_cfg.scene.entities
  assert env_cfg.actions.joint_pos.scale == env_cfg.robot.action_scale
  assert env_cfg.viewer.body_name == "torso"
  assert env_cfg.commands.twist.viz.z_offset == pytest.approx(0.5)
  assert env_cfg.events.push_robot is None
  assert env_cfg.rewards.air_time.params["sensor_names"] == ["foot_contact"]
  assert env_cfg.events.foot_friction.params["asset_cfg"].geom_names == ["foot_geom"]


def test_play_cfg_extends_episode_length(minimal_mjcf: Path):
  cfg = CustomRobotVelocityEnvCfg_PLAY(
    robot=CustomRobotAssetCfg(mjcf_path=str(minimal_mjcf))
  )
  assert cfg.episode_length_s == int(1e9)
