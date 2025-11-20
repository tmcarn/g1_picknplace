"""Minimal stand-in for the H1 MPC controller wrapper.

The real implementation solves an optimization problem; this placeholder just
computes smooth PD torques toward the standing configuration so that the demo
script can run end-to-end without the original dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class MPCState:
    qpos: jnp.ndarray
    qvel: jnp.ndarray


class MPCControllerWrapper:
    """Drop-in replacement with the same public API as the original wrapper."""

    def __init__(self, config_module):
        self.config = config_module
        self.robot_height = getattr(config_module, "robot_height", 0.85)
        self.n_joints = getattr(config_module, "n_joints", 43)
        self.kp = 800.0  # Increased from 200 for stronger stability
        self.kd = 40.0   # Increased from 10 for better damping
        self.state = MPCState(
            qpos=jnp.concatenate([config_module.p0, config_module.quat0, config_module.q0]),
            qvel=jnp.zeros(self.n_joints + 6),
        )
        self.last_tau = jnp.zeros(self.n_joints)

    def reset(self, qpos, qvel) -> None:
        self.state = MPCState(qpos=jnp.array(qpos), qvel=jnp.array(qvel))
        self.last_tau = jnp.zeros_like(self.last_tau)

    def run(
        self,
        qpos,
        qvel,
        ref_cmd,
        contact,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return joint torques plus reference joint/velocity trajectories."""

        # Desired joint trajectory is simply the standing reference
        q_des = self.config.q0
        dq_des = jnp.zeros_like(q_des)

        # Extract actuated joints from the full qpos vector (skip floating base)
        qpos = jnp.array(qpos)
        qvel = jnp.array(qvel)
        joint_pos = qpos[7 : 7 + self.n_joints]
        joint_vel = qvel[6 : 6 + self.n_joints]

        # Basic PD control on joints
        tau = self.kp * (q_des[: self.n_joints] - joint_pos[: self.n_joints]) - self.kd * joint_vel[: self.n_joints]

        # Add orientation stabilization to prevent falling
        quat_curr = np.array(qpos[3:7])  # [w, x, y, z] in MuJoCo convention
        quat_des = np.array(self.config.quat0)  # upright orientation
        
        # Compute rotation error using quaternion difference
        # For small angles, orientation error ≈ 2 * (quat_des * quat_curr_conj).xyz
        # Simplified: use the vector part directly as tilt error
        quat_error = self._quat_multiply(quat_des, self._quat_conjugate(quat_curr))
        tilt_error = quat_error[1:4]  # [x, y, z] components represent axis-angle error
        
        # Extract base angular velocity
        base_ang_vel = np.array(qvel[3:6])
        
        # Stabilization gains for torso orientation
        kp_orient = 400.0
        kd_orient = 60.0
        
        # Map orientation error to corrective joint torques
        # Pitch error (tilt_error[1]) → hip pitch joints
        # Roll error (tilt_error[0]) → hip roll joints
        pitch_correction = kp_orient * tilt_error[1] - kd_orient * base_ang_vel[1]
        roll_correction = kp_orient * tilt_error[0] - kd_orient * base_ang_vel[0]
        
        # Apply corrections to leg joints (indices from G1 model)
        # Left hip pitch (idx 0), right hip pitch (idx 6)
        tau = tau.at[0].add(pitch_correction)
        tau = tau.at[6].add(pitch_correction)
        
        # Left hip roll (idx 1), right hip roll (idx 7)
        tau = tau.at[1].add(roll_correction)
        tau = tau.at[7].add(-roll_correction)  # opposite sign for roll
        
        # Left ankle pitch (idx 4), right ankle pitch (idx 10)
        tau = tau.at[4].add(pitch_correction * 0.5)
        tau = tau.at[10].add(pitch_correction * 0.5)
        
        self.last_tau = tau

        return tau, q_des, dq_des
    
    def _quat_multiply(self, q1, q2):
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quat_conjugate(self, q):
        """Return conjugate of quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
