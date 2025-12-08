import cvxpy as cp
import mujoco
import numpy as np
from matplotlib import pyplot as plt

from g1_mpc import SimpleMPCBalance

# class WholeBodyController:
#     def __init__(self, mpc_controller:SimpleMPCBalance):
#         self.mpc = mpc_controller
        
#         # Number of actuated joints (excluding floating base)
#         self.n_joints = self.mpc.model.nu  # 43 joints for full G1
#         self.nv = self.mpc.model.nv  # Total DOFs including floating base
        
#         # WBC weights
#         self.w_contact = 1000.0      # Track desired contact forces
#         self.w_torque = 0.01         # Minimize torques
#         self.w_accel = 0.1           # Minimize accelerations
        
#         self.Kp = np.diag([200, 200, 500, 1000, 1500, 1000])
#         self.Kd = np.diag([20, 20, 30, 30, 30, 30])

#         # Weight on acceleration tracking
#         self.H = np.eye(self.nv) * 1.0

#         # Weight on force tracking  
#         self.K = np.eye(12) * 100.0

#         self.x_des = self.mpc.get_state()
#         self.x_dot_des = np.zeros(self.x_des.shape) # Rate of Change should be 0

#         self.prev_u = np.zeros(12)
#         self.prev_q_ddot = np.zeros(40)

#     def get_x_ddot_des(self, x, x_dot):
#         return self.Kp * (self.x_des - x) + self.Kd * (self.x_dot_des - x_dot)
    
#     def get_q_ddot_des(self, x, x_dot):
#         '''
#         Takes in desired x_ddot in the task space and returns 
#         the required q_ddot in the joint space.

#         Utilizes Task Heirarchy using Nullspace Projection

#         Task 1:
#         Make sure torso is oriented correcly

#         Task 2: 
#         Make sure torso is positioned correctly
        
#         :param x_ddot_des: Desired x_ddot in the task space
#         '''

#         # Initial Vaules
#         J_c = self.mpc.get_contact_jacobians()
#         N_0 = np.eye(J_c.shape) - (np.linalg.pinv(J_c) @ J_c)
#         x_ddot_des = self.get_x_ddot_des(x, x_dot)
#         x1_ddot_des = x_ddot_des[:3] # Desired CoM position
#         x2_ddot_des = x_ddot_des[3:] # Desired CoM orientation

#         q_dot = self.mpc.data.qvel.copy()
#         q_ddot_des_0 = self._compute_dynamic_pinv(J_c) @ (-J_c @ q_dot) # TODO: How to define q_dot

#         # ===============TASK 1====================
#         q_ddot_des_1, N_1 = self.iter_q_ddot_des(self.mpc.get_com_jacobian(), x1_ddot_des, q_ddot_des_0, N_0, self.mpc.get_M_inv())

#         # ===============TASK 2====================
#         q_ddot_des_2, N_2 = self.iter_q_ddot_des(self.mpc.get_omega_jacobian(), x2_ddot_des, q_ddot_des_1, N_1, self.mpc.get_M_inv())

#         return q_ddot_des_2

#     def iter_q_ddot_des(self, J_i, x_ddot_des, q_ddot_des_prev, N_prev, M_inv):
#         nv = J_i.shape[1]

#         q_dot = self.mpc.data.qvel.copy()

#         # Compute Jacobian derivative (numerical)
#         J_i_dot = np.zeros_like(J_i)  # Simplified - you can compute this

#         # Current Jacobian Projected into Prev Null Space 
#         J_i_pre = J_i @ N_prev

#         J_i_pre_dyn_inv = self.compute_dynamic_pinv(J_i_pre, M_inv)
#         J_i_pre_pinv = np.linalg.pinv(J_i_pre)

#         q_ddot_des_new =  q_ddot_des_prev + J_i_pre_dyn_inv @ (x_ddot_des - J_i_dot @ q_dot - J_i @ q_ddot_des_prev)

#         # Update Null Space
#         N_i = np.eye(nv) - (J_i_pre_pinv @ J_i_pre)
#         N_new = N_prev @ N_i

#         return q_ddot_des_new, N_new

#     def compute_dynamic_pinv(self, J, M_inv):
#         return M_inv @ J.T @ (np.linalg.inv(J @ M_inv @ J.T))
    
#     def compute_joint_torques(self, q_ddot_des, U):
#         delta_qdd = cp.Variable(self.nv) 
#         delta_u = cp.Variable(12)    
#         tau = cp.Variable(self.nv)

#         u = U[:-3]
#         F_ext = U[-3:]

#         cost = cp.quad_form(delta_qdd, self.H) + cp.quad_form(delta_qdd, self.K)

#         constraints = []

#         # Selection matrix: maps joint torques to full generalized forces
#         S_b = np.zeros((self.nv, self.nv))
#         S_b[6:, :] = np.eye(self.nv)  # Joints start at index 6 (after floating base)

#         h = self.mpc.data.qfrc_bias.copy()
#         tau_f = self.mpc.get_contact_jacobian().T @ (delta_u + u) + self.mpc.get_external_force_jacobian().T @ F_ext
#         dynamics = self.mpc.get_M() @ (delta_qdd + q_ddot_des) + h - ((S_b @ tau) + tau_f)
#         constraints.append(dynamics == 0)

#         # Torque limits
#         tau_min = self.mpc.model.actuator_ctrlrange[:, 0]
#         tau_max = self.mpc.model.actuator_ctrlrange[:, 1]
#         constraints.append(tau >= tau_min)
#         constraints.append(tau <= tau_max)

#         # Friction Cone Constraints
#         # ===== FOOT 1 CONSTRAINTS ==
#         # Normal force limits: F_min <= F1z <= F_max
#         constraints.append(u[2] >= self.F_min)
#         constraints.append(u[2] <= self.F_max)
        
#         # Friction cone: -μ*F1z <= F1x <= μ*F1z
#         constraints.append(u[0] >= -self.mu * u[2])
#         constraints.append(u[0] <= self.mu * u[2])
        
#         # Friction cone: -μ*F1z <= F1y <= μ*F1z
#         constraints.append(u[1] >= -self.mu * u[2])
#         constraints.append(u[1] <= self.mu * u[2])
        
#         # ===== FOOT 2 CONSTRAINTS =====
#         # Normal force limits: F_min <= F2z <= F_max
#         constraints.append(u[5] >= self.F_min)
#         constraints.append(u[5] <= self.F_max)
        
#         # Friction cone: -μ*F2z <= F2x <= μ*F2z
#         constraints.append(u[3] >= -self.mu * u[5])
#         constraints.append(u[3] <= self.mu * u[5])
        
#         # Friction cone: -μ*F2z <= F2y <= μ*F2z
#         constraints.append(u[4] >= -self.mu * u[5])
#         constraints.append(u[4] <= self.mu * u[5])
    
#         # Solve
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.OSQP, verbose=False)
        
#         return tau.value

        
    # def compute_q_dot_des_hier(self, U):
    #     """
    #     Task 0: CoM (highest)
    #     Task 1: Orientation (second)
    #     """

    #     # Get mass matrix
    #     M = np.zeros((self.nv, self.nv))
    #     mujoco.mj_fullM(self.model, M, self.data.qM)
    #     M_inv = np.linalg.inv(M)

    #     # Initial Vaules
    #     J_0 = self.get_contact_jacobian()
    #     J_0_dyn_inv = self.compute_dynamic_pinv(J_0, M_inv)

    #     N_0 = np.eye(J_0.shape) - (np.linalg.pinv(J_0) @ J_0)

    #     x = self.mpc.get_state()

    #     # A, B = self.mpc.state_transition_model(x)

    #     # x_dot = A @ x + B @ U

    #     x = x[:6] # [Theta, p_com]
    #     x_dot = x[6:12]

    #     x_ddot_des = self.get_x_ddot_des(x, x_dot)

    #     x1_ddot_des = x_ddot_des[3:] # Desired CoM position
    #     x2_ddot_des = x_ddot_des[:3] # Desired body orientation 
    
    #     q_dot = self.mpc.data.qvel.copy()

    #     q_ddot_des_0 = J_0_dyn_inv @ (-J_0 @ q_dot) # First priority is contact forces

    #     J_1 = self.get_com_jacobian()
    #     J_1_0 = J_1 @ N_0

    #     J_1_0_pinv = np.linalg.pinv(J_1_0)

    #     N_1_0 = np.eye(J_1_0.shape) - (J_1_0_pinv @ J_1_0) # Update Nullspace for next Task

    #     q_ddot_des_1 = q_ddot_des_0 + self.compute_dynamic_pinv(J_1_0) @ (x1_ddot_des - J_1 @ q_ddot_des_0)


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
        self.H = np.eye(self.nv) * 1000000.0
        # Weight on Force tracking  
        self.K = np.eye(12) * 100.0

        full_x_des = self.mpc.x_desired
        self.x_des = full_x_des[:6] # Theta, p_c
        self.x_dot_des = np.zeros(6) # Theta_dot, p_c_dot

        
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
        A, B = self.mpc.state_transition_model(x)
        x_dot = (A @ x) + (B @ U) # Desired change from MPC

        position = x[:6]
        velocity = x_dot[:6]
        acceleration = x_dot[6:12]

        x_ddot_des = self.get_x_ddot_des(position, velocity, acceleration)

        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M_inv = np.linalg.inv(M)

        # J_pinv = self.compute_dynamic_pinv(J_tasks, M_inv)  # (nv, 6)
        J_pinv = np.linalg.pinv(J_tasks)
        q_ddot_des = J_pinv @ x_ddot_des  # (nv, 6) @ (6, 1)

        return q_ddot_des

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
        return self.Kp @ (self.x_des - x) + self.Kd @ (self.x_dot_des - x_dot)
    
    def compute_dynamic_pinv(self, J, M_inv):
        return M_inv @ J.T @ (np.linalg.inv(J @ M_inv @ J.T))
    
    def plots(self):
        pass