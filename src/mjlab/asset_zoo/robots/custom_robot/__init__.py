"""Utility helpers for loading custom robot assets at runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, ContactSensorCfg


def _resolve_mjcf_path(path: str | None) -> Path:
  """Resolve an MJCF path from either the provided string or env variable."""
  candidate = path or os.environ.get("MJLAB_CUSTOM_ROBOT_XML", "")
  if not candidate:
    raise FileNotFoundError(
      "No MJCF path provided for custom robot. Set MJLAB_CUSTOM_ROBOT_XML or "
      "override --env.robot.mjcf-path."
    )

  resolved = Path(candidate).expanduser().resolve()
  if not resolved.is_file():
    raise FileNotFoundError(f"Custom robot MJCF not found at: {resolved}")
  return resolved


def _default_joint_map(value: float) -> dict[str, float]:
  """Create a regex mapping that targets every joint with the same value."""
  return {".*": value}


def _make_spec_loader(
  xml_path: Path,
  mesh_assets_dir: str | None,
  asset_glob: str,
  asset_recursive: bool,
) -> Callable[[], mujoco.MjSpec]:
  """Create a callable that loads an MJCF spec with bundled assets."""

  def _loader() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))

    # Collect mesh and texture assets if available.
    assets: dict[str, bytes] = {}
    meshdir = spec.meshdir if getattr(spec, "meshdir", None) else None

    if mesh_assets_dir:
      candidate_root = Path(mesh_assets_dir).expanduser().resolve()
    else:
      candidate_root = xml_path.parent / meshdir if meshdir else xml_path.parent

    if candidate_root.exists():
      update_assets(
        assets,
        candidate_root,
        meshdir,
        glob=asset_glob,
        recursive=asset_recursive,
      )
    if assets:
      spec.assets = assets

    return spec

  return _loader


@dataclass
class CustomRobotAssetCfg:
  """Configuration describing how to import a custom robot MJCF asset."""

  mjcf_path: str = ""
  """Absolute or relative path to the MJCF file describing the robot."""

  mesh_assets_dir: str | None = None
  """Optional directory containing additional assets referenced by the MJCF."""

  asset_glob: str = "*"
  """Glob used to collect assets from ``mesh_assets_dir`` (default: all files)."""

  asset_recursive: bool = True
  """Whether to recursively collect assets from subdirectories."""

  init_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
  init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  init_joint_pos: dict[str, float] | None = None
  init_joint_vel: dict[str, float] | None = None
  default_joint_pos: float = 0.0
  default_joint_vel: float = 0.0

  joint_regex: tuple[str, ...] = (".*",)
  """Regex patterns identifying joints that should be actuated."""

  effort_limit: float = 60.0
  stiffness: float = 40.0
  damping: float = 1.5
  frictionloss: float = 0.0
  armature: float = 0.0
  soft_joint_pos_limit_factor: float = 0.95

  action_scale: float = 0.5
  """Scale applied to normalized actions (mapped to joint targets)."""

  viewer_body_name: str | None = None
  """Body to follow in the viewer. Defaults to the MJCF root body."""

  command_viz_height: float | None = None
  """Optional Z offset for command visualization arrows."""

  contact_bodies: tuple[str, ...] = ()
  """Bodies that should report contacts against ``contact_body2`` (default terrain)."""

  contact_body2: str = "terrain"
  """Secondary contact body used when generating contact sensors."""

  contact_sensor_suffix: str = "_contact"
  """Suffix appended to each contact body when naming the generated sensors."""

  contact_num: int = 4
  """Maximum number of simultaneous contacts tracked per sensor."""

  contact_data: tuple[str, ...] = ("found",)
  """Which contact statistics to expose for each generated sensor."""

  contact_reduce: str = "netforce"
  """Aggregation strategy for the generated sensors."""

  foot_geom_names: tuple[str, ...] = ()
  """Geom names that should receive randomized friction (if available)."""

  pose_stds: dict[str, float] | None = None
  """Optional per-joint posture tracking standard deviations."""

  disable_push_events: bool = False
  """Disable the random push events used in humanoid tasks by default."""

  def resolve_mjcf_path(self) -> Path:
    return _resolve_mjcf_path(self.mjcf_path)


def make_custom_robot_entity_cfg(
  cfg: CustomRobotAssetCfg,
) -> tuple[EntityCfg, tuple[str, ...]]:
  """Create an :class:`EntityCfg` for a custom robot and associated sensors."""

  xml_path = cfg.resolve_mjcf_path()
  spec_loader = _make_spec_loader(
    xml_path,
    cfg.mesh_assets_dir,
    cfg.asset_glob,
    cfg.asset_recursive,
  )

  contact_sensor_cfgs: list[ContactSensorCfg] = []
  contact_sensor_names: list[str] = []
  for body in cfg.contact_bodies:
    sensor_name = f"{body}{cfg.contact_sensor_suffix}"
    contact_sensor_cfgs.append(
      ContactSensorCfg(
        name=sensor_name,
        body1=body,
        body2=cfg.contact_body2,
        num=cfg.contact_num,
        data=cfg.contact_data,  # type: ignore[arg-type]
        reduce=cfg.contact_reduce,  # type: ignore[arg-type]
      )
    )
    contact_sensor_names.append(sensor_name)

  init_state = EntityCfg.InitialStateCfg(
    pos=cfg.init_pos,
    rot=cfg.init_rot,
    joint_pos=cfg.init_joint_pos or _default_joint_map(cfg.default_joint_pos),
    joint_vel=cfg.init_joint_vel or _default_joint_map(cfg.default_joint_vel),
  )

  articulation = EntityArticulationInfoCfg(
    actuators=(
      ActuatorCfg(
        joint_names_expr=list(cfg.joint_regex),
        effort_limit=cfg.effort_limit,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
        frictionloss=cfg.frictionloss,
        armature=cfg.armature,
      ),
    ),
    soft_joint_pos_limit_factor=cfg.soft_joint_pos_limit_factor,
  )

  entity_cfg = EntityCfg(
    init_state=init_state,
    spec_fn=spec_loader,
    articulation=articulation,
    sensors=tuple(contact_sensor_cfgs),
  )

  return entity_cfg, tuple(contact_sensor_names)
