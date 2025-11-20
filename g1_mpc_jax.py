"""
G1 JAX-based MPC demo adapted from the H1 controller snippet.
Requires the `mpx` MPC stack with a compatible `config_g1` module.
"""

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))

import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
import mujoco
import mujoco.viewer

import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_g1 as config

xml_path = os.path.join(dir_path, "unitree_g1", "g1_mocap_29dof_with_hands.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mpc_frequency = 50.0
sim_frequency = 500.0
model.opt.timestep = 1 / sim_frequency

mpc = mpc_wrapper.MPCControllerWrapper(config)
data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])

tau = jnp.zeros(config.n_joints)

from timeit import default_timer as timer

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_step(model, data)
    viewer.sync()

    delay = int(0 * sim_frequency)
    print("Delay:", delay)

    mpc.robot_height = config.robot_height
    mpc.reset(data.qpos.copy(), data.qvel.copy())

    counter = 0
    while viewer.is_running():
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:
            if counter != 0:
                for _ in range(delay):
                    qpos = data.qpos.copy()
                    qvel = data.qvel.copy()
                    tau_fb = -3 * (qvel[6 : 6 + config.n_joints])
                    data.ctrl = tau + tau_fb
                    mujoco.mj_step(model, data)
                    counter += 1

            start = timer()
            ref_base_lin_vel = jnp.array([0.3, 0.0, 0.0])
            ref_base_ang_vel = jnp.array([0.0, 0.0, 0.0])

            mpc_input = np.array(
                [
                    ref_base_lin_vel[0],
                    ref_base_lin_vel[1],
                    ref_base_lin_vel[2],
                    ref_base_ang_vel[0],
                    ref_base_ang_vel[1],
                    ref_base_ang_vel[2],
                    1.0,
                ]
            )

            contact = np.zeros(config.n_contact)

            tau, q, dq = mpc.run(qpos, qvel, mpc_input, contact)
            stop = timer()
            print(f"Time elapsed: {stop - start}")

        counter += 1
        data.ctrl = tau - 3 * qvel[6 : 6 + config.n_joints]
        mujoco.mj_step(model, data)
        viewer.sync()
