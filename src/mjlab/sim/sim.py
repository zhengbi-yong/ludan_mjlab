from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import logging

import mujoco
import mujoco_warp as mjwarp
import warp as wp

from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import WarpBridge
from mjlab.utils.nan_guard import NanGuard, NanGuardCfg
from mjlab.utils.spec_config import SpecCfg


_LOGGER = logging.getLogger(__name__)
_CONTACT_SAFETY_FACTOR = 4
_CONSTRAINT_SAFETY_FACTOR = 2

# Type aliases for better IDE support while maintaining runtime compatibility
# At runtime, WarpBridge wraps the actual MJWarp objects.
if TYPE_CHECKING:
  ModelBridge = mjwarp.Model
  DataBridge = mjwarp.Data
else:
  ModelBridge = WarpBridge
  DataBridge = WarpBridge

_JACOBIAN_MAP = {
  "auto": mujoco.mjtJacobian.mjJAC_AUTO,
  "dense": mujoco.mjtJacobian.mjJAC_DENSE,
  "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}
_CONE_MAP = {
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
_INTEGRATOR_MAP = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_SOLVER_MAP = {
  "newton": mujoco.mjtSolver.mjSOL_NEWTON,
  "cg": mujoco.mjtSolver.mjSOL_CG,
  "pgs": mujoco.mjtSolver.mjSOL_PGS,
}


@dataclass
class MujocoCfg(SpecCfg):
  """Configuration for MuJoCo simulation parameters."""

  # Integrator settings.
  timestep: float = 0.002
  integrator: Literal["euler", "implicitfast"] = "implicitfast"

  # Friction settings.
  impratio: float = 1.0
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"

  # Solver settings.
  jacobian: Literal["auto", "dense", "sparse"] = "auto"
  solver: Literal["newton", "cg", "pgs"] = "newton"
  iterations: int = 100
  tolerance: float = 1e-8
  ls_iterations: int = 50
  ls_tolerance: float = 0.01

  # Other.
  gravity: tuple[float, float, float] = (0, 0, -9.81)

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    attrs = {
      "jacobian": _JACOBIAN_MAP[self.jacobian],
      "cone": _CONE_MAP[self.cone],
      "integrator": _INTEGRATOR_MAP[self.integrator],
      "solver": _SOLVER_MAP[self.solver],
      "timestep": self.timestep,
      "impratio": self.impratio,
      "gravity": self.gravity,
      "iterations": self.iterations,
      "tolerance": self.tolerance,
      "ls_iterations": self.ls_iterations,
      "ls_tolerance": self.ls_tolerance,
    }
    for k, v in attrs.items():
      setattr(spec.option, k, v)


@dataclass(kw_only=True)
class SimulationCfg:
  nconmax: int | None = None
  njmax: int | None = None
  ls_parallel: bool = True  # Boosts perf quite noticeably.
  mujoco: MujocoCfg = field(default_factory=MujocoCfg)
  nan_guard: NanGuardCfg = field(default_factory=NanGuardCfg)


def _resolve_buffer_capacities(
  *,
  num_envs: int,
  mj_model: mujoco.MjModel,
  mj_data: mujoco.MjData,
  cfg_nconmax: int | None,
  cfg_njmax: int | None,
) -> tuple[int, int]:
  """Determine contact and constraint capacities for a multi-world simulation."""

  per_env_contacts = max(1, int(mj_data.ncon))
  estimated_nconmax = max(
    512, per_env_contacts * num_envs * _CONTACT_SAFETY_FACTOR
  )
  nconmax = (
    estimated_nconmax if cfg_nconmax is None else max(cfg_nconmax, estimated_nconmax)
  )

  per_env_constraints = max(1, int(mj_data.nefc))
  estimated_njmax = max(
    5,
    per_env_constraints * _CONSTRAINT_SAFETY_FACTOR,
    int(getattr(mj_model, "njmax", 0)),
  )
  njmax = (
    estimated_njmax if cfg_njmax is None else max(cfg_njmax, estimated_njmax)
  )

  return nconmax, njmax


class Simulation:
  """GPU-accelerated MuJoCo simulation powered by MJWarp."""

  def __init__(
    self, num_envs: int, cfg: SimulationCfg, model: mujoco.MjModel, device: str
  ):
    self.cfg = cfg
    self.device = device
    self.wp_device = wp.get_device(self.device)
    self.num_envs = num_envs

    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    nconmax, njmax = _resolve_buffer_capacities(
      num_envs=self.num_envs,
      mj_model=self._mj_model,
      mj_data=self._mj_data,
      cfg_nconmax=self.cfg.nconmax,
      cfg_njmax=self.cfg.njmax,
    )

    with wp.ScopedDevice(self.wp_device):
      self._wp_model = mjwarp.put_model(self._mj_model)
      self._wp_model.opt.ls_parallel = cfg.ls_parallel

      if self.cfg.nconmax is not None and nconmax > self.cfg.nconmax:
        per_env = max(1, int(self._mj_data.ncon))
        _LOGGER.info(
          "Expanding nconmax from %d to %d for %d environments (≈%d contacts/env)",
          self.cfg.nconmax,
          nconmax,
          self.num_envs,
          per_env,
        )
      if self.cfg.njmax is not None and njmax > self.cfg.njmax:
        per_env_nefc = max(1, int(self._mj_data.nefc))
        _LOGGER.info(
          "Expanding njmax from %d to %d for %d environments (≈%d constraints/env)",
          self.cfg.njmax,
          njmax,
          self.num_envs,
          per_env_nefc,
        )

      self._wp_data = mjwarp.put_data(
        self._mj_model,
        self._mj_data,
        nworld=self.num_envs,
        nconmax=nconmax,
        njmax=njmax,
      )

    self._model_bridge = WarpBridge(self._wp_model, nworld=self.num_envs)
    self._data_bridge = WarpBridge(self._wp_data)

    self.use_cuda_graph = self.wp_device.is_cuda and wp.is_mempool_enabled(
      self.wp_device
    )
    self.create_graph()

    self.nan_guard = NanGuard(cfg.nan_guard, self.num_envs, self._mj_model)

  def create_graph(self) -> None:
    self.step_graph = None
    self.forward_graph = None
    if self.use_cuda_graph:
      with wp.ScopedCapture() as capture:
        mjwarp.step(self.wp_model, self.wp_data)
      self.step_graph = capture.graph
      with wp.ScopedCapture() as capture:
        mjwarp.forward(self.wp_model, self.wp_data)
      self.forward_graph = capture.graph

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def wp_model(self) -> mjwarp.Model:
    return self._wp_model

  @property
  def wp_data(self) -> mjwarp.Data:
    return self._wp_data

  @property
  def data(self) -> "DataBridge":
    return cast("DataBridge", self._data_bridge)

  @property
  def model(self) -> "ModelBridge":
    return cast("ModelBridge", self._model_bridge)

  # Methods.

  def expand_model_fields(self, fields: list[str]) -> None:
    """Expand model fields to support per-environment parameters."""
    invalid_fields = [f for f in fields if not hasattr(self._mj_model, f)]
    if invalid_fields:
      raise ValueError(f"Fields not found in model: {invalid_fields}")

    expand_model_fields(self._wp_model, self.num_envs, fields)

  def reset(self) -> None:
    # TODO(kevin): Should we be doing anything here?
    pass

  def forward(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.forward_graph is not None:
        wp.capture_launch(self.forward_graph)
      else:
        mjwarp.forward(self.wp_model, self.wp_data)

  def step(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      with self.nan_guard.watch(self.data):
        if self.use_cuda_graph and self.step_graph is not None:
          wp.capture_launch(self.step_graph)
        else:
          mjwarp.step(self.wp_model, self.wp_data)

  def close(self) -> None:
    pass
