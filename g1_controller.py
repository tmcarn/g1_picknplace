import mujoco

from g1_wbc import WholeBodyController
from g1_mpc import SimpleMPCBalance

mpc = SimpleMPCBalance(mpc_frq=50)
wbc = WholeBodyController(mpc)

for i in range(10_000):
    u = mpc.compute_optimal_control() # Determine Optimal Contact Forces

    tau = wbc.compute_optimal_tau(u) # Determine Optimal Joint Torque from Contact Forces
    
    mpc.step(tau)
    mpc.render()