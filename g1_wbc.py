import cvxpy as cp
import mujoco
import numpy as np
from matplotlib import pyplot as plt

from g1_mpc import SimpleMPCBalance

class SimpleWBC:
    def __init__(self, mpc:SimpleMPCBalance):
        self.mpc = mpc
        self.model = mpc.model
        self.data = mpc.data
        self.nv = mpc.model.nv
        self.n_joints = mpc.model.nu

        # Gains for Task Space tracking
        self.Kp = np.diag([1000, 1500, 1000, 200, 200, 500]) * 0.01
        self.Kd = np.diag([30, 30, 30, 20, 20, 30]) * 0.1

        # Weight on Acceleration tracking
        self.H = np.eye(self.nv) * 10_000.0
        # Weight on Force tracking  
        self.K = np.eye(12) * 1.0

        full_x_des = self.mpc.x_desired
        self.x_des = full_x_des[:6] # Theta, p_c
        self.x_dot_des = np.zeros(6) # Theta_dot, p_c_dot
        self.x_ddot_des = np.zeros(6)

        self.prev_com_vel = None

        
    def compute_torques(self, U):
        """
        Simplest WBC: Solve dynamics for torques
        
        Args:
            q_ddot_des: Desired joint accelerations (nv,)
            F_contacts: Contact forces [F1, F2, M1, M2] (12,)
        
        Returns:
            tau: Joint torques (n_joints,)
        """
        
        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        
        # Get bias forces (Coriolis + gravity)
        h = self.data.qfrc_bias.copy()

        delta_qdd, delta_u = self.qp_relaxation(U)

        F_contacts = U[:-3] # Ignores F_ext (dummy variable)
        F_ext = U[-3:]

        # Add Mx back to Contact Forces
        F_contacts = np.concatenate([
                                        U[0:3],              # Left foot forces [fx1, fy1, fz1]
                                        U[3:6],              # Right foot forces [fx2, fy2, fz2]
                                        [0.0], U[6:8],       # Left foot moments [0, my1, mz1]
                                        [0.0], U[8:10]       # Right foot moments [0, my2, mz2]
                                    ])
        
        F_contacts += delta_u
        
        # Contact forces contribute: J^T·F
        J_contact = self.get_contact_jacobian()
        tau_contacts = J_contact.T @ F_contacts

        # External forces contribute: J_e^T F_ext
        J_external = self.get_external_force_jacobian()
        tau_external = J_external.T @ F_ext


        U[:6] = U[:6] + delta_u[:6]
        U[6:8] = U[6:8] + delta_u[7:9]
        U[8:10] = U[8:10] + delta_u[11:12]

        q_ddot_des = self.compute_q_dot_des(U) + delta_qdd

        tau = (M @ q_ddot_des) + h - tau_contacts - tau_external 
        tau = tau[6:]
        
        # Clip to actuator limits
        tau_min = self.model.actuator_ctrlrange[:, 0]
        tau_max = self.model.actuator_ctrlrange[:, 1]
        tau = np.clip(tau, tau_min, tau_max)
        
        return tau
    
    def get_contact_jacobian(self):
        nv = self.nv
        
        jac_left_trans = np.zeros((3, nv))
        jac_left_rot = np.zeros((3, nv))
        jac_right_trans = np.zeros((3, nv))
        jac_right_rot = np.zeros((3, nv))
        
        left_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
        mujoco.mj_jacSite(self.model, self.data, jac_left_trans, jac_left_rot, left_site)
        mujoco.mj_jacSite(self.model, self.data, jac_right_trans, jac_right_rot, right_site)
        
        J_contact = np.vstack([jac_left_trans, jac_right_trans, jac_left_rot, jac_right_rot])
       
        return J_contact
   
    def get_external_force_jacobian(self):
        """
        Get Jacobian for external force application point (box CoM)
        
        Returns:
            J_e: (3, nv) Jacobian mapping joint velocities to box CoM velocity
        """
        nv = self.nv
        
        # Get box body
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        
        # Compute Jacobian at box CoM
        jac_box_trans = np.zeros((3, nv))
        jac_box_rot = np.zeros((3, nv))  # Not needed for pure force, only for moments
        
        # Use body Jacobian (at body's CoM)
        mujoco.mj_jacBodyCom(self.model, self.data, jac_box_trans, jac_box_rot, box_body_id)
        
        # For external force, we only need translational Jacobian
        J_e = jac_box_trans
        
        return J_e
    
    def get_com_jacobian(self):
        """
        CoM position Jacobian
        Maps: q̇ → ṗ_c (linear velocity of CoM)
        """
        nv = self.model.nv
        jac_com = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac_com, 0)
    
        return jac_com 
    
    def compute_q_dot_des(self, U):
        # Get Jacobians
        J_com = self.get_com_jacobian() 
        J_orient = self.mpc.get_orientation_jacobian() 
        
        # Stack tasks
        J_tasks = np.vstack([J_orient, J_com]) 

        x = self.mpc.get_state()
        A_c, B_c = self.mpc.state_transition_model(x)

        # Convert to Discrete Time
        A_d = np.eye(A_c.shape[0]) + (A_c * self.mpc.dt)
        B_d = B_c * self.mpc.dt

        x_dot = (A_d @ x) + (B_d @ U) # Desired change from MPC

        position = x[:6]
        velocity = x_dot[:6]
        acceleration = x_dot[6:12]

        # x_ddot_des = self.get_x_ddot_des(position, velocity, acceleration)
        self.x_ddot_des = acceleration

        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M_inv = np.linalg.inv(M)

        J_pinv = self.compute_dynamic_pinv(J_tasks, M_inv)  # (nv, 6)
        q_ddot_des = J_pinv @ self.x_ddot_des  # (nv, 6) @ (6, 1)

        return q_ddot_des
    
    def get_com_acceleration(self):
        """
        Compute CoM linear and angular acceleration.
        
        Parameters:
        - model: MuJoCo model
        - data: MuJoCo data
        
        Returns:
        - com_accel: (6,) array [linear_xyz, angular_xyz] in m/s² and rad/s²
        """
        # Get CoM linear velocity
        # mj_comVel computes velocity and stores it in data.subtree_linvel[0]
        mujoco.mj_comVel(self.model, self.data)
        com_linear_vel = self.data.subtree_linvel[0].copy()
        
        # Get CoM angular velocity (angular momentum / total mass)
        # For simplicity, we'll use the floating base angular velocity
        # or compute from angular momentum if needed
        
        # Method 1: Use floating base angular velocity (for humanoid)
        if self.model.nq >= 7:  # Has floating base (quaternion)
            com_angular_vel = self.data.qvel[3:6].copy()  # Angular velocity from floating base
        else:
            com_angular_vel = np.zeros(3)
        
        # Combine into task velocity
        com_vel = np.hstack([com_linear_vel, com_angular_vel])
        
        # Compute acceleration via finite difference
        if self.prev_com_vel is not None:
            com_accel = (com_vel - self.prev_com_vel) / self.mpc.dt
        else:
            com_accel = np.zeros(6)
        
        # Update previous velocity
        self.prev_com_vel = com_vel.copy()
        
        return com_accel

    def qp_relaxation(self, U):
        delta_qdd = cp.Variable(self.nv) 
        delta_u = cp.Variable(12)    
        tau = cp.Variable(self.n_joints)

        F_contacts = U[:-3]

        # Add Mx back to Contact Forces
        F_contacts = np.concatenate([
                                        U[0:3],              # Left foot forces [fx1, fy1, fz1]
                                        U[3:6],              # Right foot forces [fx2, fy2, fz2]
                                        [0.0], U[6:8],       # Left foot moments [0, my1, mz1]
                                        [0.0], U[8:10]       # Right foot moments [0, my2, mz2]
                                    ])
        F_ext = U[-3:]

        cost = cp.quad_form(delta_qdd, self.H) + cp.quad_form(delta_u, self.K)

        constraints = []

        h = self.mpc.data.qfrc_bias.copy()

        u_total = delta_u + F_contacts

        q_ddot_des = self.compute_q_dot_des(U)
        q_ddot_total = delta_qdd + q_ddot_des

        # Contact forces contribute: J^T·F
        J_contact = self.get_contact_jacobian()
        tau_contacts = J_contact.T @ u_total

        # External forces contribute: J_e^T F_ext
        J_external = self.get_external_force_jacobian()
        tau_external = J_external.T @ F_ext

        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        S_b = np.zeros((self.nv, self.n_joints))
        S_b[6:, :] = np.eye(self.n_joints)  

        dynamics = (M @ q_ddot_total) + h - tau_contacts - tau_external - (S_b @ tau)
        constraints.append(dynamics == 0)

        mu = 0.7        # Friction coefficient
        F_min = 25.0    # Minimum normal force
        F_max = 500.0   # Maximum normal force
        M_max = 50.0    # Maximum moment
    
        # LEFT FOOT
        # Normal force bounds
        constraints.append(u_total[2] >= F_min)
        constraints.append(u_total[2] <= F_max)
        
        # Friction cone (tangential forces)
        constraints.append(u_total[0] <= mu * u_total[2])
        constraints.append(u_total[0] >= -mu * u_total[2])
        constraints.append(u_total[1] <= mu * u_total[2])
        constraints.append(u_total[1] >= -mu * u_total[2])
        
        # Moment limits
        constraints.append(u_total[6] >= -M_max)   # mx1 = 0
        constraints.append(u_total[6] <= M_max)
        constraints.append(u_total[7] >= -M_max)   # my1
        constraints.append(u_total[7] <= M_max)
        constraints.append(u_total[8] >= -M_max)   # mz1
        constraints.append(u_total[8] <= M_max)
        
        # RIGHT FOOT
        # Normal force bounds
        constraints.append(u_total[5] >= F_min)
        constraints.append(u_total[5] <= F_max)
        
        # Friction cone
        constraints.append(u_total[3] <= mu * u_total[5])
        constraints.append(u_total[3] >= -mu * u_total[5])
        constraints.append(u_total[4] <= mu * u_total[5])
        constraints.append(u_total[4] >= -mu * u_total[5])
        
        # Moment limits
        constraints.append(u_total[9] >= -M_max)
        constraints.append(u_total[9] <= M_max)
        constraints.append(u_total[10] >= -M_max)
        constraints.append(u_total[10] <= M_max)
        constraints.append(u_total[11] >= -M_max)
        constraints.append(u_total[11] <= M_max)
    

        # Torque limit Constraints
        tau_min = self.mpc.model.actuator_ctrlrange[:, 0]
        tau_max = self.mpc.model.actuator_ctrlrange[:, 1]
        constraints.append(tau >= tau_min)
        constraints.append(tau <= tau_max)

        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        print(f"delta_qdd max:{np.max(np.abs(delta_qdd.value))}, delta_u max:{np.max(np.abs(delta_u.value))}")
        return delta_qdd.value, delta_u.value

    def get_x_ddot_des(self, x, x_dot, x_ddot):
        return x_ddot + self.Kp @ (self.x_des - x) + self.Kd @ (self.x_dot_des - x_dot)
    
    def compute_dynamic_pinv(self, J, M_inv):
        return M_inv @ J.T @ (np.linalg.inv(J @ M_inv @ J.T))
    
    def plots(self, q_ddot_des):
        pass