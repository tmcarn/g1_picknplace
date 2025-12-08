import mujoco

from g1_wbc import SimpleWBC
from g1_mpc import SimpleMPCBalance
import numpy as np

mpc = SimpleMPCBalance(mpc_frq=50)
wbc = SimpleWBC(mpc)



def log_step(U, tau):
    print(f"\n{'='*80}")
    print(f"Step {i:5d} | Time: {mpc.data.time:6.2f}s")
    print(f"{'='*80}")
    
    # Contact Forces
    print(f"Contact Forces (MPC Output):")
    print(f"  Left Foot  - F: [{U[0]:7.2f}, {U[1]:7.2f}, {U[2]:7.2f}] N  |  M: [{U[6]:6.2f}, {U[7]:6.2f}] Nm")
    print(f"  Right Foot - F: [{U[3]:7.2f}, {U[4]:7.2f}, {U[5]:7.2f}] N  |  M: [{U[8]:6.2f}, {U[9]:6.2f}] Nm")
    print(f"  Total Fz: {U[2] + U[5]:7.2f} N  (should be ~{mpc.box_mass * 9.81 + 36 * 9.81:.1f} N)")
    
    # Joint Torques (show first few joints)
    print(f"\nJoint Torques (WBC Output):")
    print(f"  Legs    - Hip:   [{tau[0]:6.2f}, {tau[1]:6.2f}, {tau[2]:6.2f}] Nm (L)  |  [{tau[6]:6.2f}, {tau[7]:6.2f}, {tau[8]:6.2f}] Nm (R)")
    print(f"          - Knee:  [{tau[3]:6.2f}] Nm (L)  |  [{tau[9]:6.2f}] Nm (R)")
    print(f"          - Ankle: [{tau[4]:6.2f}, {tau[5]:6.2f}] Nm (L)  |  [{tau[10]:6.2f}, {tau[11]:6.2f}] Nm (R)")
    print(f"  Waist   - [{tau[12]:6.2f}, {tau[13]:6.2f}, {tau[14]:6.2f}] Nm (yaw, roll, pitch)")
    print(f"  Max torque: {np.max(np.abs(tau)):6.2f} Nm")
    
    # Robot State
    state = mpc.get_state()
    print(f"\nRobot State:")
    print(f"  Orientation - Roll: {np.degrees(state[0]):6.2f}°, Pitch: {np.degrees(state[1]):6.2f}°, Yaw: {np.degrees(state[2]):6.2f}°")
    print(f"  CoM Position - [{state[3]:6.3f}, {state[4]:6.3f}, {state[5]:6.3f}] m")

for i in range(10_000):
    U = mpc.compute_optimal_control() # Determine Optimal Contact Forces
    tau = wbc.compute_torques(U) # Determine Optimal Joint Torque from Contact Forces
    
    if i%1 == 0:
        log_step(U, tau)

    mpc.step_tau(tau)

    mpc.render()

