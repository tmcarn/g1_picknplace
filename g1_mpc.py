import numpy as np
import mujoco
import mujoco.viewer
import cvxpy as cp
from scipy.spatial.transform import Rotation as R

import os
from collections import defaultdict
import time


class SimpleMPCBalance:
    def __init__(self, mpc_frq=50, render_mode="human", render_fps=30):
        self.xml_path = "unitree_g1/g1_with_box_torque_ctrl.xml"
        
        # Select a graphics backend for the viewer
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Get dimensions
        # self.nu = self.model.nu  # Number of actuators
        # self.nq = self.model.nq  # Number of position coordinates
        # self.nv = self.model.nv  # Number of velocity coordinates

        # MPC Parameters
        self.mpc_frq = mpc_frq
        self.dt = 1 / self.mpc_frq
        self.horizon = 20 # steps

        # State weights (from paper)
        self.Q = np.diag([
            1500,
            2000,
            1000,
            1000, 
            1000,
            1000,
            1, 
            3, 
            10, 
            1, 
            1, 
            1, 
            1, 
            1, 
            1
        ])
        
        # Decrease R to allow more aggressive control
        self.R = np.diag([
            1, # F1
            1, 
            1, 
            1, # F2
            1,
            1,
            5, # M1
            5, 
            5, # M2
            5, 
            0, # F_ext
            0,
            0
        ]) * 10e-4

        # Variable Dimensions
        self.nx = 15 # number of state variables (Theta, p_c, omega, p_c_dot, g)
        self.nu = 13 # number of control variable (F1, F2, M1, M2, Fext)

        self.weights = {
            'com_position': 3000.0,    # Keep CoM over Support Polygon
            'orientation': 2000.0,     # Stay upright
            'velocity': 100.0,         # Minimize velocities (stability)
            'angular_momentum': 50.0,  # Don't spin
            'torque': 0.1,            # Minimize effort
            'smoothness': 1.0,        # Smooth controls
        }

        # Load the standing keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.data.qpos[:] = self.model.key_qpos[keyframe_id]
        self.data.qvel[:] = self.model.key_qvel[keyframe_id]
        mujoco.mj_forward(self.model, self.data)
        print("Loaded in 'stand' keyframe as initial position")

        # Initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Previous control for smoothness
        self.u_prev = np.zeros(self.nu)

        # Environmental Constants
        self.norm_vec = np.array([0, 0, 1])
        self.gravity = np.array([0,0,-9.81])
        self.box_mass = 1 #kg

        # Use initial state as desired state
        self.x_desired = self.get_state()
        print(self.x_desired)

        # Constraint Parameters
        self.mu = 0.7
        self.F_min = 25.0   # Minimum normal force (N) - keeps contact
        self.F_max = 500.0  # Maximum normal force (N) - robot/ground limits
        self.M_max = 50.0   # Maximum moment (N⋅m)

        # Render 
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

        self.print_model_info()
        self.contact_data = self.get_contacts()

        self.reset_noise = 0.01
        self.reset()

    def state_transition_model(self, x):
        '''
        x: [theta, p_c, omega, p_c_dot, g] (g is dummie variable)
        u: [F_1, F_2, M_1, M_2, F_ext] shape: (13,)

        Input: x --> (15,) 
        Output: dx/dt --> (15,)
        '''
        theta = x[:3]
        roll, pitch, yaw = theta

        A = np.zeros((15,15))

        c_theta = np.cos(pitch)
        s_theta = np.sin(pitch)
        c_psi = np.cos(yaw)
        s_psi = np.sin(yaw)
        
        # Roll Not Included because M_x is ignored
        R_b = np.array([
            [c_theta * c_psi,  -s_psi,  0],
            [c_theta * s_psi,   c_psi,  0],
            [-s_theta,          0,      1]
        ])

        A[:3, 6:9] = R_b
        A[3:6, 9:12] = np.eye(3)
        A[9:12, 12:] = np.eye(3)

        B = np.zeros((15,13))

        # Linear Forces Equation (3, 13)
        F = np.zeros((3,13))
        F[:,:3] = np.eye(3)
        F[:,3:6] = np.eye(3)
        F = F/self.box_mass
        
        
        # Moment Equations
        M = np.zeros((3, 13))

        L = np.array([[0,0],
                      [1,0],
                      [0,1]])
        
        distance_vectors = self.get_distance_vectors()
        
        r1x = self.skew_symmetric(distance_vectors[0])
        r2x = self.skew_symmetric(distance_vectors[1])
        r_ex = self.skew_symmetric(distance_vectors[2])

        I_G = self.get_inertia_tensor()
        I_G_inv = np.linalg.inv(I_G)

        M[:, :3] = I_G_inv @ r1x      # Effect of F1
        M[:, 3:6] = I_G_inv @ r2x     # Effect of F2
        M[:, 10:] = I_G_inv @ r_ex    # Effect of F_ext
        M[:, 6:8] = I_G_inv @ L       # Effect of M1
        M[:, 8:10] = I_G_inv @ L      # Effect of M2

        # Add to Matrix
        B[6:9, :] = M      
        B[9:12, :] = F     

        # Linearized State Transition Function
        return A, B
    
    def compute_optimal_control(self):
        x0 = self.get_state()

        A, B = self.state_transition_model(x0)

        # Decision variables for each step in horizon
        x = [cp.Variable(15) for _ in range(self.horizon + 1)] # Included initial state
        u = [cp.Variable(13) for _ in range(self.horizon)]

        # Cost Function
        cost = 0
        constraints = []

        # required_fz = 45 * 9.81

        # Initial state constraint
        constraints.append(x[0] == x0)

        for k in range(self.horizon):
            # Cost
            x_error = x[k] - self.x_desired
            cost += cp.quad_form(x_error, self.Q)
            cost += cp.quad_form(u[k], self.R)

            # total_fz = u[k][2] + u[k][5]
            # cost += 100000.0 * cp.square(total_fz - required_fz)

            # ===== DYNAMIC CONSTRAINTS =====
            x_dot = A @ x[k] + B @ u[k]
            constraints.append(x[k+1] == x[k] + self.dt * x_dot)

            # ===== FOOT 1 CONSTRAINTS =====
            # Normal force limits: F_min <= F1z <= F_max
            constraints.append(u[k][2] >= self.F_min)
            constraints.append(u[k][2] <= self.F_max)
            
            # Friction cone: -μ*F1z <= F1x <= μ*F1z
            constraints.append(u[k][0] >= -self.mu * u[k][2])
            constraints.append(u[k][0] <= self.mu * u[k][2])
            
            # Friction cone: -μ*F1z <= F1y <= μ*F1z
            constraints.append(u[k][1] >= -self.mu * u[k][2])
            constraints.append(u[k][1] <= self.mu * u[k][2])
            
            # ===== FOOT 2 CONSTRAINTS =====
            # Normal force limits: F_min <= F2z <= F_max
            constraints.append(u[k][5] >= self.F_min)
            constraints.append(u[k][5] <= self.F_max)
            
            # Friction cone: -μ*F2z <= F2x <= μ*F2z
            constraints.append(u[k][3] >= -self.mu * u[k][5])
            constraints.append(u[k][3] <= self.mu * u[k][5])
            
            # Friction cone: -μ*F2z <= F2y <= μ*F2z
            constraints.append(u[k][4] >= -self.mu * u[k][5])
            constraints.append(u[k][4] <= self.mu * u[k][5])

            # ===== EXTERNAL FORCE CONSTRAINTS =====
            constraints.append(u[k][10:] == self.gravity * self.box_mass)

        
        # Terminal cost
        x_error_final = x[self.horizon] - self.x_desired
        cost += cp.quad_form(x_error_final, self.Q * 10) # Terminal cost is emphasized
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve(solver=cp.OSQP, verbose=False)

        return u[0].value
    
    # def compute_joint_torques_from_forces(self, F_desired):
    #     """
    #     Convert desired contact forces to joint torques using inverse dynamics
    #     """
    #     nv = self.model.nv  # 6 (floating base) + 12 (legs) + ... = total DOFs
        
    #     # Get contact Jacobians (translational AND rotational)
    #     jac_left_trans = np.zeros((3, nv))
    #     jac_left_rot = np.zeros((3, nv))
    #     jac_right_trans = np.zeros((3, nv))
    #     jac_right_rot = np.zeros((3, nv))
        
    #     left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    #     right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
    #     # Get both translational and rotational Jacobians
    #     mujoco.mj_jacSite(self.model, self.data, jac_left_trans, jac_left_rot, left_foot_site)
    #     mujoco.mj_jacSite(self.model, self.data, jac_right_trans, jac_right_rot, right_foot_site)
        
    #     # Extract forces and moments (6D wrenches)
    #     F_left = F_desired[:3]      # [fx, fy, fz]
    #     M_left = F_desired[3:6]     # [mx, my, mz]
    #     F_right = F_desired[6:9]    # [fx, fy, fz]
    #     M_right = F_desired[9:12]   # [mx, my, mz]
        
    #     print(f"F_left: {F_left}, M_left: {M_left}")
    #     print(f"F_right: {F_right}, M_right: {M_right}")

    #     print(f"qfrc_bias (all): {self.data.qfrc_bias}")
    #     print(f"qfrc_bias legs [6:18]: {self.data.qfrc_bias[6:18]}")
    #     print(f"Robot mass: {np.sum(self.model.body_mass)} kg")
    #     print(f"Base height: {self.data.qpos[2]} m")
        
    #     # Compute torques from contact forces AND moments
    #     tau_contacts = (jac_left_trans.T @ F_left + jac_left_rot.T @ M_left + 
    #                     jac_right_trans.T @ F_right + jac_right_rot.T @ M_right)
        
    #     # Add gravity and Coriolis compensation
    #     tau_total = self.data.qfrc_bias.copy() - tau_contacts
        
    #     # Extract leg joint torques (skip 6 DOF floating base)
    #     tau_legs = tau_total[6:18]
        
    #     print(f"Computed torques: {tau_legs}")
    #     print(f"Max torque: {np.max(np.abs(tau_legs)):.2f} Nm")
        
    #     # Clip to actuator limits
    #     tau_legs = np.clip(tau_legs, 
    #                     [-88, -139, -88, -139, -50, -50,
    #                         -88, -139, -88, -139, -50, -50],
    #                     [88, 139, 88, 139, 50, 50,
    #                         88, 139, 88, 139, 50, 50])
        
    #     return tau_legs
        
    # def compute_joint_torques_from_forces(self, F_desired):
    #     """
    #     Compute feedforward torques from MPC forces, plus PD stabilization
    #     """
    #     mujoco.mj_forward(self.model, self.data)
        
    #     nv = self.model.nv
        
    #     # Get Jacobians
    #     jac_left_trans = np.zeros((3, nv))
    #     jac_right_trans = np.zeros((3, nv))
        
    #     left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    #     right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
    #     mujoco.mj_jacSite(self.model, self.data, jac_left_trans, None, left_foot_site)
    #     mujoco.mj_jacSite(self.model, self.data, jac_right_trans, None, right_foot_site)
        
    #     F_left = F_desired[:3]
    #     F_right = F_desired[6:9]
        
    #     # Feedforward from contact forces
    #     tau_ff = jac_left_trans.T @ F_left + jac_right_trans.T @ F_right
    #     tau_ff_legs = tau_ff[6:18]
        
    #     # PD feedback to stabilize around standing pose
    #     q_des = self.init_qpos[7:19]  # Desired leg joint positions
    #     q_curr = self.data.qpos[7:19]  # Current leg joint positions
    #     qd_curr = self.data.qvel[6:18]  # Current leg joint velocities
        
    #     kp = 100.0  # Position gain
    #     kd = 20.0   # Velocity gain
        
    #     tau_fb = kp * (q_des - q_curr) - kd * qd_curr
        
    #     # Total torque = feedforward + feedback
    #     tau_total = tau_ff_legs + tau_fb
        
    #     print(f"tau_ff: {tau_ff_legs[:3]}")
    #     print(f"tau_fb: {tau_fb[:3]}")
    #     print(f"tau_total: {tau_total[:3]}")
        
    #     tau_total = np.clip(tau_total,
    #                         [-88, -139, -88, -139, -50, -50,
    #                         -88, -139, -88, -139, -50, -50],
    #                         [88, 139, 88, 139, 50, 50,
    #                         88, 139, 88, 139, 50, 50])
        
    #     return tau_total
        
    # def compute_joint_torques_from_forces(self, F_desired):
    #     """
    #     Proper inverse dynamics using MuJoCo
    #     """
    #     # Update forward dynamics first
    #     mujoco.mj_forward(self.model, self.data)
        
    #     # Apply external forces at feet
    #     left_foot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    #     right_foot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        
    #     # Extract forces and moments
    #     F_left = F_desired[:3]
    #     M_left = F_desired[3:6]
    #     F_right = F_desired[6:9]
    #     M_right = F_desired[9:12]
        
    #     # Apply as external forces (in world frame)
    #     # xfrc_applied is [fx, fy, fz, mx, my, mz] per body
    #     self.data.xfrc_applied[left_foot_body] = np.concatenate([F_left, M_left])
    #     self.data.xfrc_applied[right_foot_body] = np.concatenate([F_right, M_right])

    #     box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    #     box_weight = np.array([0, 0, self.box_mass * -9.81, 0, 0, 0])  # 5kg downward
        
    #     self.data.xfrc_applied[box_body] = box_weight
        
    #     # Compute required actuator forces using inverse dynamics
    #     # This computes: M(q)*qacc + C(q,qvel) = tau + J^T*F_ext
    #     # Solving for tau given qacc=0 and F_ext
    #     mujoco.mj_inverse(self.model, self.data)
        
    #     # qfrc_inverse now contains required generalized forces
    #     tau_required = self.data.qfrc_inverse.copy()
        
    #     # Extract leg and waist joint torques
    #     tau_all = tau_required[6:]
        
    #     print(f"Applied forces - Left: {F_left}, Right: {F_right}")
    #     print(f"qfrc_inverse legs: {tau_all}")
    #     print(f"Max torque: {np.max(np.abs(tau_all)):.2f} Nm")
        
    #     # Clear external forces for next iteration
    #     self.data.xfrc_applied[:] = 0
        
    #     return tau_all   
    
    # def compute_joint_torques_from_forces(self, F_desired):
    #     """
    #     Compute feedforward torques from contact forces + PD stabilization
    #     """
    #     mujoco.mj_forward(self.model, self.data)
        
    #     nv = self.model.nv
        
    #     # Get foot Jacobians
    #     jac_left_trans = np.zeros((3, nv))
    #     jac_left_rot = np.zeros((3, nv))
    #     jac_right_trans = np.zeros((3, nv))
    #     jac_right_rot = np.zeros((3, nv))
        
    #     left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    #     right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
    #     mujoco.mj_jacSite(self.model, self.data, jac_left_trans, jac_left_rot, left_foot_site)
    #     mujoco.mj_jacSite(self.model, self.data, jac_right_trans, jac_right_rot, right_foot_site)
        
    #     # Extract forces
    #     F_left = F_desired[:3]
    #     M_left = F_desired[3:6]
    #     F_right = F_desired[6:9]
    #     M_right = F_desired[9:12]
        
    #     # Feedforward torques from contact forces
    #     tau_contacts = (jac_left_trans.T @ F_left + jac_left_rot.T @ M_left + 
    #                     jac_right_trans.T @ F_right + jac_right_rot.T @ M_right)
        
    #     # Gravity compensation
    #     tau_gravity = self.data.qfrc_bias.copy()
        
    #     # Feedforward component
    #     tau_ff = -tau_gravity + tau_contacts
    #     tau_ff_actuated = tau_ff[6:]
        
    #     # PD Feedback to stabilize around keyframe pose
    #     q_des = self.init_qpos[7:]  # Desired joint positions (all joints)
    #     q_curr = self.data.qpos[7:]  # Current joint positions
    #     qd_curr = self.data.qvel[6:]  # Current joint velocities
        
    #     # Different gains for different joint groups
    #     kp = np.zeros(self.model.nu)
    #     kd = np.zeros(self.model.nu)
        
    #     # Legs (0-11): Moderate stiffness
    #     kp[0:12] = 50.0
    #     kd[0:12] = 10.0
        
    #     # Waist (12-14): Lower stiffness to allow MPC to move it
    #     kp[12:15] = 20.0
    #     kd[12:15] = 5.0
        
    #     # Arms (15+): Higher stiffness to hold box
    #     kp[15:] = 100.0
    #     kd[15:] = 20.0
        
    #     # PD feedback torques
    #     tau_fb = kp * (q_des - q_curr) - kd * qd_curr
        
    #     # Total = feedforward + feedback
    #     tau_total = tau_ff_actuated + tau_fb
        
    #     print(f"Waist FF: {tau_ff_actuated[12:15]}, FB: {tau_fb[12:15]}, Total: {tau_total[12:15]}")
        
    #     # Clip to limits
    #     ctrl_range = self.model.actuator_ctrlrange.copy()
    #     tau_total = np.clip(tau_total, ctrl_range[:, 0], ctrl_range[:, 1])
        
    #     return tau_total
        
    def get_state(self):
        x = np.zeros(15)

        # Theta
        torso_quat = self.data.qpos[3:7]  # Quaternion [w, x, y, z] in MuJoCo

        rotation = R.from_quat(torso_quat, scalar_first=True)
        roll, pitch, yaw = rotation.as_euler('ZYX', degrees=False)
        
        x[0:3] = [roll, pitch, yaw]
        
        # p_c
        p_c = self.get_g1_com()
        x[3:6] = p_c
        
        # omega
        omega = self.data.qvel[3:6]  # Body frame angular velocity
        x[6:9] = omega

        # p_c_dot
        p_c_dot = self.compute_robot_com_velocity()
        x[9:12] = p_c_dot

        # g
        x[12:] = self.gravity

        return x

    def compute_robot_com_velocity(self):
        """
        Compute robot COM velocity excluding the box
        
        Returns:
            p_c_dot: robot COM velocity [vx, vy, vz]
        """
        total_mass = 0.0
        weighted_vel = np.zeros(3)
        
        for body_id in range(self.model.nbody):
            # Skip world body and box
            if body_id == 0 or body_id == "box":
                continue
            
            body_mass = self.model.body_mass[body_id]
            
            # Get body velocity
            # MuJoCo computes body velocities in data after forward kinematics
            body_vel = self.data.cvel[body_id, 3:6]
            
            weighted_vel += body_mass * body_vel
            total_mass += body_mass
        
        robot_com_vel = weighted_vel / total_mass
        return robot_com_vel
        
    def get_inertia_tensor(self):
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        
        # Extract rotational inertia (first 3x3 block corresponds to angular DOFs)
        # For floating base: DOFs are [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, joint1, ...]
        I_G = M[3:6, 3:6]  # Angular part of mass matrix
        
        return I_G
    
    def get_distance_vectors(self):
        p_c = self.get_g1_com()

        contact_dict = self.get_contacts()


        f1_contact_points = contact_dict["left_ankle_roll_link"]["positions"]

        if len(f1_contact_points) > 0:
            # Has contacts - use mean
            f1_center_point = np.mean(f1_contact_points, axis=0)
        else:
            # No contacts - use body position as fallback
            left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
            f1_center_point = self.data.xpos[left_foot_id].copy()

        r1 = f1_center_point - p_c # Moment arm from CoM to contact point 1
        

        f2_contact_points = contact_dict["right_ankle_roll_link"]["positions"]

        if len(f2_contact_points) > 0:
            # Has contacts - use mean
            f2_center_point = np.mean(f2_contact_points, axis=0)
        else:
            # No contacts - use body position as fallback
            right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
            f2_center_point = self.data.xpos[right_foot_id].copy()

        r2 = f2_center_point - p_c # Moment arm from CoM to contact point 2

        f_e_center_point = self.get_obj_com()
        r_e = f_e_center_point - p_c

        return np.array([r1, r2, r_e])

    def skew_symmetric(self, v):
        """
        Create skew-symmetric matrix from 3D vector
        
        Args:
            v: array-like, shape (3,)
        
        Returns:
            v_cross: ndarray, shape (3, 3)
        """
        return np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])
    
    def get_contacts(self):
        contact_data = defaultdict(lambda: {
        'positions': [],
        'forces': [],
        'moments':[],
        'num_contacts': 0,
        })

        # Check all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
                
            # Get bodies in contact
            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            body_1name = self.model.body(body1).name
            body_2name = self.model.body(body2).name

            if body_1name == 'world':
                link_name = body_2name

            elif body_2name == 'world':
                link_name = body_1name

            else: # Only concerned with Foot - World Contacts
                continue

            # Get force vector
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_lin = force[:3]
            moment = force[3:]

            # Transform to world
            contact_frame = contact.frame.reshape(3, 3)
            force_world = contact_frame @ force_lin
            moment_world = contact_frame @ moment
            force_mag = np.linalg.norm(force_world)
            moment_mag = np.linalg.norm(moment_world)

            # Skip Weak Contacts
            if force_mag > 25 or moment_mag > 10: # Newtons
                # Accumulate data
                contact_data[link_name]['positions'].append(contact.pos.copy())
                contact_data[link_name]['forces'].append(force_world)
                contact_data[link_name]['moments'].append(moment_world)
                contact_data[link_name]['num_contacts'] += 1

        return contact_data
           
    def print_model_info(self):
        # Print model info
        print(f"\n{'='*50}")
        print(f"Model: {self.xml_path}")
        print(f"{'='*50}")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of DOF: {self.model.nv}")
        print(f"Number of actuators: {self.model.nu}")
        print(f"Number of links: {self.model.nbody}")

        # Print joint names and their indices
        print(f"\nJoint Information:")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = self.model.jnt_type[i]
            qpos_addr = self.model.jnt_qposadr[i]
            print(f"  Joint {i}: {joint_name} (type: {joint_type}, qpos index: {qpos_addr})")

        # Print Link Names and their indices
        print(f"\nLink Information:")
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"  Body {i}: {body_name}")

        print(f"{'='*50}")
        # Print Physics Info 
        print(f"Physics timestep: {self.model.opt.timestep} seconds")
        print(f"Physics frequency: {1/self.model.opt.timestep} Hz")
        # Print Controls Info
        self.frame_skip = int((1 / self.model.opt.timestep) / self.mpc_frq)
        print(f"Frame skip: {self.frame_skip}")
        print(f"Control frequency: {1/(self.model.opt.timestep * self.frame_skip)} Hz")

    def get_obj_com(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        p_o = self.data.xpos[box_id]
        return p_o

    def get_g1_com(self):
        """
        Compute full robot COM excluding the carried box
        """
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        
        total_mass = 0.0
        weighted_pos = np.zeros(3)
        
        # Iterate through all bodies
        for body_id in range(self.model.nbody):
            # Skip world body (id=0) and box body
            if body_id == 0 or body_id == box_id:
                continue
            
            body_mass = self.model.body_mass[body_id]
            body_pos = self.data.xpos[body_id]
            
            weighted_pos += body_mass * body_pos
            total_mass += body_mass
        
        robot_com = weighted_pos / total_mass
        return robot_com

    def get_com_jacobian(self):
        """
        CoM position Jacobian
        Maps: q̇ → ṗ_c (linear velocity of CoM)
        """
        nv = self.model.nv
        jac_com = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac_com, 0)
    
        return jac_com 

    def get_omega_jacobian(self):
        pass

    def get_contact_jacobian(self):
        """
        Get stacked contact Jacobian for both feet
        Returns J_contact: (12, nv) for [F1, F2, M1, M2]
        """
        nv = self.nv
        
        # Foot Jacobians
        jac_left_trans = np.zeros((3, nv))
        jac_left_rot = np.zeros((3, nv))
        jac_right_trans = np.zeros((3, nv))
        jac_right_rot = np.zeros((3, nv))
        
        left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
        mujoco.mj_jacSite(self.model, self.data, jac_left_trans, jac_left_rot, left_foot_site)
        mujoco.mj_jacSite(self.model, self.data, jac_right_trans, jac_right_rot, right_foot_site)
        
        # Stack: [F1(3), F2(3), M1(3), M2(3)] = 12D
        J_contact = np.vstack([
            jac_left_trans,   # F1
            jac_right_trans,  # F2
            jac_left_rot,     # M1
            jac_right_rot     # M2
        ])
        
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

    
    def get_M_inv(self):
        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M_inv = np.linalg.inv(M)
        return M_inv

    def get_M(self):
        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def step(self, tau):
        """Take a step in the environment"""

        # Clear forces from previous step
        self.data.qfrc_applied[:] = 0

        # Set tau for each actuator
        self.data.ctrl[:] = tau
        
        # Step through physics multiple times with same action (frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None):
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to initial state with small random perturbations
        self.data.qpos[:] = self.init_qpos + np.random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nq
        )
        self.data.qvel[:] = self.init_qvel + np.random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nv
        )
        
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Sleep to match desired render FPS
            time.sleep(self.render_dt)
            self.viewer.sync()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            if self.render_mode == "human":
                self.viewer.close()
            self.viewer = None
