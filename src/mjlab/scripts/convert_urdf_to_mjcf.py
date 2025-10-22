"""CLI helper to convert URDF assets into MJCF for mjlab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import os
import tyro

from urdf2mjcf.convert import convert_urdf_to_mjcf


@dataclass(frozen=True)
class ConvertUrdfConfig:
  """Configuration for converting a URDF file into MJCF."""

  urdf_path: Path = Path("ludan_v0/urdf/ludan_v0.urdf")
  """Path to the source URDF file."""

  output_path: Path = Path("ludan_v0/mjcf/ludan_v0.xml")
  """Destination for the generated MJCF file."""

  package_map: tuple[str, ...] = ("ludan_v0=./ludan_v0",)
  """Mappings of ROS-style package URIs (``package://name``) to filesystem roots.

  Each entry should use ``name=path`` syntax. Relative paths are resolved from
  the current working directory.
  """

  keep_resolved_urdf: bool = False
  """Whether to keep the intermediate URDF with resolved asset paths."""

  strip_actuators: bool = True
  """Remove any ``<actuator>`` blocks from the generated MJCF (recommended)."""


def _parse_package_map(entries: tuple[str, ...]) -> Dict[str, Path]:
  mapping: Dict[str, Path] = {}
  for entry in entries:
    if "=" not in entry:
      raise ValueError(f"Invalid package mapping '{entry}'. Expected format name=path.")
    name, path = entry.split("=", 1)
    if not name:
      raise ValueError(f"Package mapping '{entry}' is missing a package name.")
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
      raise FileNotFoundError(
        f"Package mapping for '{name}' points to missing directory: {resolved_path}"
      )
    mapping[name] = resolved_path
  return mapping


def _resolve_package_uris(text: str, mapping: Dict[str, Path], base_dir: Path) -> str:
  resolved = text
  for name, path in mapping.items():
    rel_path = Path(os.path.relpath(path, base_dir)).as_posix()
    resolved = resolved.replace(f"package://{name}/", rel_path + "/")
  return resolved


def _strip_actuator_sections(text: str) -> Tuple[str, bool]:
  """Remove any ``<actuator>`` sections from an MJCF document."""

  lines = text.splitlines()
  kept_lines: list[str] = []
  skipping = False
  removed = False

  for line in lines:
    if not skipping and "<actuator" in line:
      removed = True
      if "</actuator>" in line:
        continue
      skipping = True
      continue

    if skipping:
      if "</actuator>" in line:
        skipping = False
      continue

    kept_lines.append(line)

  if skipping:
    raise ValueError("Encountered unterminated <actuator> block while stripping actuators.")

  new_text = "\n".join(kept_lines)
  if text.endswith("\n"):
    new_text += "\n"

  return new_text, removed


def run(cfg: ConvertUrdfConfig) -> None:
  package_map = _parse_package_map(cfg.package_map)

  urdf_text = cfg.urdf_path.read_text()
  resolved_text = _resolve_package_uris(urdf_text, package_map, cfg.urdf_path.parent)

  resolved_urdf_path = cfg.urdf_path.with_name(cfg.urdf_path.stem + "_resolved.urdf")
  resolved_urdf_path.write_text(resolved_text)

  cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
  convert_urdf_to_mjcf(resolved_urdf_path, cfg.output_path)

  if cfg.strip_actuators:
    mjcf_text = cfg.output_path.read_text()
    stripped_text, removed = _strip_actuator_sections(mjcf_text)
    if removed:
      cfg.output_path.write_text(stripped_text)

  if not cfg.keep_resolved_urdf:
    resolved_urdf_path.unlink(missing_ok=True)

  print(f"Converted {cfg.urdf_path} -> {cfg.output_path}")


def main() -> None:
  cfg = tyro.cli(ConvertUrdfConfig)
  run(cfg)


if __name__ == "__main__":
  main()
