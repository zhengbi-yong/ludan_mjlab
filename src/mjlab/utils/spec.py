"""MjSpec utils."""

import mujoco


def get_non_free_joints(spec: mujoco.MjSpec) -> tuple[mujoco.MjsJoint, ...]:
  """Returns all joints except the free joint."""
  joints: list[mujoco.MjsJoint] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(jnt)
  return tuple(joints)


def get_free_joint(spec: mujoco.MjSpec) -> mujoco.MjsJoint | None:
  """Returns the free joint. None if no free joint exists."""
  joint: mujoco.MjsJoint | None = None
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      joint = jnt
      break
  return joint


def disable_collision(geom: mujoco.MjsGeom) -> None:
  """Disables collision for a geom."""
  geom.contype = 0
  geom.conaffinity = 0


def is_joint_limited(jnt: mujoco.MjsJoint) -> bool:
  """Returns True if a joint is limited."""
  match jnt.limited:
    case mujoco.mjtLimited.mjLIMITED_TRUE:
      return True
    case mujoco.mjtLimited.mjLIMITED_AUTO:
      return jnt.range[0] < jnt.range[1]
    case _:
      return False
