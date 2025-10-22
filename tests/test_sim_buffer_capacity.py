import sys
import types
from types import SimpleNamespace

import pytest

if "warp" not in sys.modules:
  sys.modules["warp"] = types.SimpleNamespace(
    config=types.SimpleNamespace(enable_backward=False, quiet=False)
  )

pytest.importorskip("mujoco")

from mjlab.sim.sim import _resolve_buffer_capacities


def make_dummy(ncon: int, nefc: int, njmax: int):
  model = SimpleNamespace(njmax=njmax)
  data = SimpleNamespace(ncon=ncon, nefc=nefc)
  return model, data


def test_capacity_scales_with_envs_when_contacts_exceed_budget():
  model, data = make_dummy(ncon=12, nefc=150, njmax=300)
  nconmax, njmax = _resolve_buffer_capacities(
    num_envs=4096,
    mj_model=model,
    mj_data=data,
    cfg_nconmax=140_000,
    cfg_njmax=300,
  )
  assert nconmax == 12 * 4096 * 4  # safety factor applied
  assert njmax == 300


def test_capacity_defaults_when_config_unspecified():
  model, data = make_dummy(ncon=0, nefc=0, njmax=200)
  nconmax, njmax = _resolve_buffer_capacities(
    num_envs=1024,
    mj_model=model,
    mj_data=data,
    cfg_nconmax=None,
    cfg_njmax=None,
  )
  assert nconmax == 1024 * 4  # fallback to 1 contact with safety factor
  assert njmax == 200  # falls back to model bound


def test_capacity_respects_constraint_growth():
  model, data = make_dummy(ncon=4, nefc=600, njmax=300)
  nconmax, njmax = _resolve_buffer_capacities(
    num_envs=256,
    mj_model=model,
    mj_data=data,
    cfg_nconmax=3000,
    cfg_njmax=500,
  )
  assert nconmax == 3000
  assert njmax == 600 * 2
