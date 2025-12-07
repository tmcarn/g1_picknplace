import cvxpy as cp
import mujoco
import numpy as np

from g1_mpc import SimpleMPCBalance

class WholeBodyController:
    def __init__(self, mpc_controller:SimpleMPCBalance):
        self.mpc = mpc_controller
        
        # Number of actuated joints (excluding floating base)
        self.n_joints = self.mpc.model.nu  # 43 joints for full G1
        self.nv = self.mpc.model.nv  # Total DOFs including floating base
        
        # WBC weights
        self.w_contact = 1000.0      # Track desired contact forces
        self.w_torque = 0.01         # Minimize torques
        self.w_accel = 0.1           # Minimize accelerations
        
        self.Kp = np.diag([200, 200, 500, 1000, 1500, 1000])
        self.Kd = np.diag([20, 20, 30, 30, 30, 30])

        # Weight on acceleration tracking
        self.H = np.eye(self.nv) * 1.0

        # Weight on force tracking  
        self.K = np.eye(12) * 100.0

        self.x_des = self.mpc.get_state()
        self.x_dot_des = np.zeros(self.x_des.shape) # Rate of Change should be 0

        self.prev_u = np.zeros(12)
        self.prev_q_ddot = np.zeros(40)

    def get_x_ddot_des(self, x, x_dot):
        return self.Kp * (self.x_des - x) + self.Kd * (self.x_dot_des - x_dot)
    
    def get_q_ddot_des(self, x, x_dot):
        '''
        Takes in desired x_ddot in the task space and returns 
        the required q_ddot in the joint space.

        Utilizes Task Heirarchy using Nullspace Projection

        Task 1:
        Make sure torso is oriented correcly

        Task 2: 
        Make sure torso is positioned correctly
        
        :param x_ddot_des: Desired x_ddot in the task space
        '''

        # Initial Vaules
        J_c = self.mpc.get_contact_jacobians()
        N_0 = np.eye(J_c.shape) - (np.linalg.pinv(J_c) @ J_c)
        x_ddot_des = self.get_x_ddot_des(x, x_dot)
        x1_ddot_des = x_ddot_des[:3] # Desired CoM position
        x2_ddot_des = x_ddot_des[3:] # Desired CoM orientation

        q_dot = self.mpc.data.qvel.copy()
        q_ddot_des_0 = self._compute_dynamic_pinv(J_c) @ (-J_c @ q_dot) # TODO: How to define q_dot

        # ===============TASK 1====================
        q_ddot_des_1, N_1 = self.iter_q_ddot_des(self.mpc.get_com_jacobian(), x1_ddot_des, q_ddot_des_0, N_0, self.mpc.get_M_inv())

        # ===============TASK 2====================
        q_ddot_des_2, N_2 = self.iter_q_ddot_des(self.mpc.get_omega_jacobian(), x2_ddot_des, q_ddot_des_1, N_1, self.mpc.get_M_inv())

        return q_ddot_des_2

    def iter_q_ddot_des(self, J_i, x_ddot_des, q_ddot_des_prev, N_prev, M_inv):
        nv = J_i.shape[1]

        q_dot = self.mpc.data.qvel.copy()

        # Compute Jacobian derivative (numerical)
        J_i_dot = np.zeros_like(J_i)  # Simplified - you can compute this

        # Current Jacobian Projected into Prev Null Space 
        J_i_pre = J_i @ N_prev

        J_i_pre_dyn_inv = self.compute_dynamic_pinv(J_i_pre, M_inv)
        J_i_pre_pinv = np.linalg.pinv(J_i_pre)

        q_ddot_des_new =  q_ddot_des_prev + J_i_pre_dyn_inv @ (x_ddot_des - J_i_dot @ q_dot - J_i @ q_ddot_des_prev)

        # Update Null Space
        N_i = np.eye(nv) - (J_i_pre_pinv @ J_i_pre)
        N_new = N_prev @ N_i

        return q_ddot_des_new, N_new

    def compute_dynamic_pinv(self, J, M_inv):
        return M_inv @ J.T @ (np.linalg.inv(J @ M_inv @ J.T))
    
    def compute_joint_torques(self, q_ddot_des, U):
        delta_qdd = cp.Variable(self.nv) 
        delta_u = cp.Variable(12)    
        tau = cp.Variable(self.nv)

        u = U[:-3]
        F_ext = U[-3:]

        cost = cp.quad_form(delta_qdd, self.H) + cp.quad_form(delta_qdd, self.K)

        constraints = []

        # Selection matrix: maps joint torques to full generalized forces
        S_b = np.zeros((self.nv, self.nv))
        S_b[6:, :] = np.eye(self.nv)  # Joints start at index 6 (after floating base)

        h = self.mpc.data.qfrc_bias.copy()
        tau_f = self.mpc.get_contact_jacobian().T @ (delta_u + u) + self.mpc.get_external_force_jacobian().T @ F_ext
        dynamics = self.mpc.get_M() @ (delta_qdd + q_ddot_des) + h - ((S_b @ tau) + tau_f)
        constraints.append(dynamics == 0)

        # Torque limits
        tau_min = self.mpc.model.actuator_ctrlrange[:, 0]
        tau_max = self.mpc.model.actuator_ctrlrange[:, 1]
        constraints.append(tau >= tau_min)
        constraints.append(tau <= tau_max)

        # Friction Cone Constraints
        # ===== FOOT 1 CONSTRAINTS ==
        # Normal force limits: F_min <= F1z <= F_max
        constraints.append(u[2] >= self.F_min)
        constraints.append(u[2] <= self.F_max)
        
        # Friction cone: -μ*F1z <= F1x <= μ*F1z
        constraints.append(u[0] >= -self.mu * u[2])
        constraints.append(u[0] <= self.mu * u[2])
        
        # Friction cone: -μ*F1z <= F1y <= μ*F1z
        constraints.append(u[1] >= -self.mu * u[2])
        constraints.append(u[1] <= self.mu * u[2])
        
        # ===== FOOT 2 CONSTRAINTS =====
        # Normal force limits: F_min <= F2z <= F_max
        constraints.append(u[5] >= self.F_min)
        constraints.append(u[5] <= self.F_max)
        
        # Friction cone: -μ*F2z <= F2x <= μ*F2z
        constraints.append(u[3] >= -self.mu * u[5])
        constraints.append(u[3] <= self.mu * u[5])
        
        # Friction cone: -μ*F2z <= F2y <= μ*F2z
        constraints.append(u[4] >= -self.mu * u[5])
        constraints.append(u[4] <= self.mu * u[5])
    
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        return tau.value

class SimpleWBC:
    def __init__(self, mpc:SimpleMPCBalance):
        self.mpc = mpc
        self.model = mpc.model
        self.data = mpc.data
        self.nv = mpc.model.nv
        self.n_joints = mpc.model.nu
        
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
        # M = np.zeros((self.nv, self.nv))
        # mujoco.mj_fullM(self.model, M, self.data.qM)
        
        # Get bias forces (Coriolis + gravity)
        h = self.data.qfrc_bias.copy()

        F_contacts = U[:-3] # Ignores F_ext (dummy variable)
        F_ext = U[-3:]

        # Add Mx back to Contact Forces
        F_contacts = np.concatenate([
                                        U[0:3],              # Left foot forces [fx1, fy1, fz1]
                                        U[3:6],              # Right foot forces [fx2, fy2, fz2]
                                        [0.0], U[6:8],       # Left foot moments [0, my1, mz1]
                                        [0.0], U[8:10]       # Right foot moments [0, my2, mz2]
                                    ])
        
        # Contact forces contribute: J^T·F
        J_contact = self._get_contact_jacobian()
        tau_contacts = J_contact.T @ F_contacts

        # External forces contribute: J_e^T F_ext
        J_external = self.get_external_force_jacobian()
        tau_external = J_external.T @ F_ext

        # q_ddot_des = self.mpc.calculate_q_ddot_des()

            
        # Solve dynamics for joint torques:
        # M·q̈_des + h = S_b·τ + J^T·F
        # S_b·τ = M·q̈_des + h - J^T·F
        
        # # Selection matrix (maps joint torques to full state)
        # S_b = np.zeros((self.nv, self.n_joints))
        # S_b[6:, :] = np.eye(self.n_joints)
        
        tau = -tau_contacts - tau_external + h
        tau = tau[6:]
        
        # Clip to actuator limits
        tau_min = self.model.actuator_ctrlrange[:, 0]
        tau_max = self.model.actuator_ctrlrange[:, 1]
        tau = np.clip(tau, tau_min, tau_max)
        
        return tau
    
    def _get_contact_jacobian(self):
        nv = self.nv
        
        jac_left_trans = np.zeros((3, nv))
        jac_left_rot = np.zeros((3, nv))
        jac_right_trans = np.zeros((3, nv))
        jac_right_rot = np.zeros((3, nv))
        
        left_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
        mujoco.mj_jacSite(self.model, self.data, jac_left_trans, jac_left_rot, left_site)
        mujoco.mj_jacSite(self.model, self.data, jac_right_trans, jac_right_rot, right_site)

            # DIAGNOSTIC
        print(f"\nJacobian columns for left ankle pitch (index 10):")
        print(f"  jac_left_trans[:, 10] = {jac_left_trans[:, 10]}")
        print(f"\nJacobian columns for left hip pitch (index 6):")
        print(f"  jac_left_trans[:, 6] = {jac_left_trans[:, 6]}")
        
        
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