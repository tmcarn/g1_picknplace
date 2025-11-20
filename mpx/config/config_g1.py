"""G1 configuration for the placeholder MPC wrapper.

This mirrors the structure of the H1 config used by the original MPC demo.
It loads the standing keyframe from the Unitree G1 MuJoCo model so callers
can initialize the simulator consistently.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import mujoco

# Resolve repository root relative to this file
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_XML_PATH = os.path.join(_REPO_ROOT, "unitree_g1", "g1_mocap_29dof_with_hands.xml")

# Load the model once so we can pull out reference poses
_MODEL = mujoco.MjModel.from_xml_path(_XML_PATH)
_KEYFRAME_ID = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_KEY, "stand")
_QPOS_REF = _MODEL.key_qpos[_KEYFRAME_ID].copy()

# Split the reference pose into floating-base position, orientation, and joint angles
p0 = jnp.array(_QPOS_REF[:3])
quat0 = jnp.array(_QPOS_REF[3:7])
q0 = jnp.array(_QPOS_REF[7:])

# Robot/solver parameters expected by the MPC controller wrapper
n_joints = _MODEL.nu
n_contact = 4
mpc_frequency = 50.0
robot_height = float(_QPOS_REF[2])

# Reasonable default joint limits / gains for future use
joint_position_limits = jnp.stack((
    jnp.full(n_joints, -3.14),
    jnp.full(n_joints, 3.14),
))

joint_velocity_limits = jnp.full(n_joints, 10.0)

def describe() -> str:
    """Return a short summary string for debugging/logging."""
    return (
        f"G1 config: nq={_MODEL.nq}, nv={_MODEL.nv}, nu={_MODEL.nu}, "
        f"height={robot_height:.3f} m"
    )
