import mujoco

from g1_wbc import SimpleWBC
from g1_mpc import SimpleMPCBalance
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

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
    des_state = mpc.x_desired
    print(f"\nRobot State:")
    print(f"  Orientation - Roll: {np.degrees(state[0]):6.2f}°, Pitch: {np.degrees(state[1]):6.2f}°, Yaw: {np.degrees(state[2]):6.2f}°")
    print(f"  CoM Position - [{state[3]:6.3f}, {state[4]:6.3f}, {state[5]:6.3f}] m")
    print(f"  Desired CoM Position - [{des_state[3]:6.3f}, {des_state[4]:6.3f}, {des_state[5]:6.3f}] m")
    
    state_hist.append(state)
    contact_forces.append(U)
    x_ddot_des_hist.append(wbc.x_ddot_des)
    x_ddot_hist.append(wbc.get_com_acceleration())

state_hist = []
contact_forces = []
x_ddot_des_hist = []
x_ddot_hist = []
sim_time = 2.5 # seconds

num_steps = int(mpc.mpc_frq * sim_time)
for i in range(num_steps):
    U = mpc.compute_optimal_control() # Determine Optimal Contact Forces
    tau = wbc.compute_torques(U) # Determine Optimal Joint Torque from Contact Forces
    
    log_step(U, tau)

    mpc.step_tau(tau)
    # mpc.step_cf(U)

    mpc.render()


def plot_orientation(orientation, des_orientation, time):
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    col = [0, 1, 2]
    labels = ["Roll", "Pitch", "Yaw"]
    colors_actual = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    colors_desired = ['#aec7e8', '#ffbb78', '#98df8a']  # Lighter versions

    for i, (col,label) in enumerate(zip(col,labels)):
        # Plot desired and actual
        ax.plot(time, des_orientation[:, col], '--', linewidth=2, 
                color=colors_desired[i], label=f'{label} Desired', alpha=0.8)
        ax.plot(time, orientation[:, col], '-', linewidth=2.5, 
                color=colors_actual[i], label=f'{label} Actual')
        
    # Styling
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angle (deg)', fontsize=12)
    ax.set_title('G1 Free Body Joint: RPY Tracking', fontsize=14, fontweight='bold')
    ax.set_ylim(-180, 180)
    ax.legend(loc='best', ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("RPY Plot.png")

def plot_com(com, com_des, time):
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    col = [0, 1, 2]
    labels = ["X", "Y", "Z"]
    colors_actual = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    colors_desired = ['#aec7e8', '#ffbb78', '#98df8a']  # Lighter versions

    for i, (col,label) in enumerate(zip(col,labels)):
        # Plot desired and actual
        ax.plot(time, com_des[:, col], '--', linewidth=2, 
                color=colors_desired[i], label=f'{label} Desired', alpha=0.8)
        ax.plot(time, com[:, col], '-', linewidth=2.5, 
                color=colors_actual[i], label=f'{label} Actual')
        
    # Styling
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Meters (m)', fontsize=12)
    ax.set_title('G1 CoM Tracking', fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("CoM Plot.png")

def plot_xddot(x_ddot, x_ddot_des, time):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    col = [0, 1, 2]
    labels = ["X", "Y", "Z"]
    colors_actual = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    colors_desired = ['#aec7e8', '#ffbb78', '#98df8a']  # Lighter versions

    for i, (col,label) in enumerate(zip(col,labels)):
        # Plot desired and actual
        ax[0].plot(time, x_ddot_des[:, col], '-', linewidth=2, 
                color=colors_desired[i], label=f'{label} Desired')
    
    col = [3, 4, 5]   
    for i, (col,label) in enumerate(zip(col,labels)):
        # Plot desired and actual
        ax[1].plot(time, x_ddot_des[:, col], '-', linewidth=2, 
                color=colors_desired[i], label=f'{label} Desired')
        
    # Styling
    ax[0].set_xlabel('Time (s)', fontsize=12)
    ax[0].set_ylabel(f'Acceleration ($m/s^2$)', fontsize=12)
    ax[0].set_title('G1 Desired Task Space Linear Acceleration', fontsize=14, fontweight='bold')
    ax[0].legend(loc='best', ncol=2, fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Styling
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel(f'Acceleration ($m/s^2$)', fontsize=12)
    ax[1].set_title('G1 Desired Task Space Angular Acceleration', fontsize=14, fontweight='bold')
    ax[1].legend(loc='best', ncol=2, fontsize=10)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Xddot Plot.png")

def plot_contact_forces(contact_forces, time):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    col = [0, 1, 2]
    labels = ["X", "Y", "Z"]
    colors_desired = ['#aec7e8', '#ffbb78', '#98df8a']  # Lighter versions

    for i, (col,label) in enumerate(zip(col,labels)):
        # Plot desired and actual
        ax[0].plot(time, contact_forces[:, col], '-', linewidth=2, 
                color=colors_desired[i], label=f'{label} Desired')
    
    col = [6, 7]
    labels = ["Y", "Z"]
    colors_desired = ['#ffbb78', '#98df8a']  # Lighter versions 
    for i, (col,label) in enumerate(zip(col,labels)):
        # Plot desired and actual
        ax[1].plot(time, contact_forces[:, col], '-', linewidth=2, 
                color=colors_desired[i], label=f'{label} Desired')
        
    # Styling
    ax[0].set_xlabel('Time (s)', fontsize=12)
    ax[0].set_ylabel(f'Force (N)', fontsize=12)
    ax[0].set_title('G1 Desired Contact Forces', fontsize=14, fontweight='bold')
    ax[0].legend(loc='best', ncol=2, fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Styling
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel(f'Moment (Nm)', fontsize=12)
    ax[1].set_title('G1 Desired Contact Moments', fontsize=14, fontweight='bold')
    ax[1].legend(loc='best', ncol=2, fontsize=10)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Contact Forces Plot.png")


# Plot State vs Desired State
state_hist = np.array(state_hist)

n_steps = state_hist.shape[0]
time = np.arange(n_steps) * mpc.dt

des_state = mpc.x_desired
des_orientation = np.tile(np.degrees(des_state[:3]), (n_steps, 1))
des_com = np.tile(des_state[3:6], (n_steps, 1))

orientation = np.degrees(state_hist[:, 0:3])
com = state_hist[:, 3:6]

plot_orientation(orientation, des_orientation, time)
plot_com(com, des_com, time)

# Plot Task Acc vs Des Task Acc
x_ddot_des_hist = np.array(x_ddot_des_hist)
x_ddot_hist = np.array(x_ddot_hist)
plot_xddot(x_ddot_hist, x_ddot_des_hist, time)

contact_forces = np.array(contact_forces)
plot_contact_forces(contact_forces, time)







