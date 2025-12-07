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

        self.data.qvel[:] = 0 

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
        self.box_mass = 2 #kg

        # Use initial state as desired state
        self.x_desired = self.get_state()
        print(self.x_desired)

        # Constraint Parameters
        self.mu = 0.7
        self.F_min = 0.0   # Minimum normal force (N) - keeps contact
        self.F_max = 500.0  # Maximum normal force (N) - robot/ground limits
        self.M_max = 9.0   # Maximum moment (N⋅m)

        # Render 
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

        self.print_model_info()
        self.contact_data = self.get_contacts()

        self.reset_noise = 0.01

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

        required_fz = 45 * 9.81

        # Initial state constraint
        constraints.append(x[0] == x0)

        for k in range(self.horizon):
            # Cost
            x_error = x[k] - self.x_desired
            cost += cp.quad_form(x_error, self.Q)
            cost += cp.quad_form(u[k], self.R)

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

            # ===== MOMENT BOUNDING CONSTRAINTS =====
            constraints.append(u[k][6] >= -self.M_max)
            constraints.append(u[k][6] <= self.M_max)
            constraints.append(u[k][7] >= -self.M_max)
            constraints.append(u[k][7] <= self.M_max)
            constraints.append(u[k][8] >= -self.M_max)
            constraints.append(u[k][8] <= self.M_max)
            constraints.append(u[k][9] >= -self.M_max)
            constraints.append(u[k][9] <= self.M_max)



        
        # TERMINAL COST
        x_error_final = x[self.horizon] - self.x_desired
        cost += cp.quad_form(x_error_final, self.Q * 10) # Terminal cost is emphasized
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve(solver=cp.OSQP, verbose=False)

        print("MPC Complete")

        return u[0].value
    
    def calculate_q_ddot_des(self):
        x = self.get_state()
        
        theta_curr = x[0:3]
        p_com_curr = x[3:6]
        omega_curr = x[6:9]
        p_com_dot_curr = x[9:12]
        
        # Desired state (from MPC)
        theta_des = self.x_desired[0:3]
        p_com_des = self.x_desired[3:6]
        
        # MUCH LOWER GAINS - Your gains are WAY too high!
        Kp_com = 5.0       # Down from 100
        Kd_com = 10.0      # Down from 50
        Kp_orient = 2.0    # Down from 50
        Kd_orient = 5.0    # Down from 20
        
        # Errors (for debugging)
        e_com = p_com_des - p_com_curr
        e_theta = theta_des - theta_curr
        
        # PD control (zero desired velocities)
        p_ddot_des = Kp_com * e_com - Kd_com * p_com_dot_curr
        omega_ddot_des = Kp_orient * e_theta - Kd_orient * omega_curr
        
        # Add joint damping
        q_dot = self.data.qvel.copy()
        joint_damping = -20.0 * q_dot  # Increased from -10
        
        # Full state
        q_ddot_des = np.zeros(self.model.nv)
        q_ddot_des[0:3] = p_ddot_des
        q_ddot_des[3:6] = omega_ddot_des
        q_ddot_des += joint_damping
        
        # CRITICAL: CLIP TO PREVENT EXPLOSION
        max_accel = 2.0  # rad/s² or m/s²
        q_ddot_des = np.clip(q_ddot_des, -max_accel, max_accel)
        
        print(f"COM Error: {e_com}")
        print(f"Theta Error: {e_theta}")
        print(f"COM_VEL Error: {p_com_dot_curr}")
        print(f"Omega Error: {omega_curr}")
        
        return q_ddot_des
    
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

    def step_tau(self, tau):
        """Take a step in the environment"""

        # Clear forces from previous step
        self.data.qfrc_applied[:] = 0

        # Set tau for each actuator
        self.data.ctrl[:] = tau
        
        # Step through physics multiple times with same action (frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def step_cf(self, U):
        """Take a step in the environment"""

        # Clear forces from previous step
        self.data.qfrc_applied[:] = 0

        # Set tau=0 for each actuator
        self.data.ctrl[:] = 0

        # Apply forces DIRECTLY to feet (no WBC)
        self.apply_contact_force(U)
        
        # Step through physics multiple times with same action (frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def apply_contact_force(self, U):
        # Apply forces DIRECTLY to feet (no WBC)
        left_foot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        right_foot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        
        # Convert U to full wrenches
        F_left = np.concatenate([U[0:3], [0], U[6:8]])   # [fx, fy, fz, 0, my, mz]
        F_right = np.concatenate([U[3:6], [0], U[8:10]]) # [fx, fy, fz, 0, my, mz]
        
        # Apply external forces
        self.data.xfrc_applied[left_foot_body] = F_left
        self.data.xfrc_applied[right_foot_body] = F_right
        
        # Zero joint torques
        self.data.ctrl[:] = 0

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
            # time.sleep(self.render_dt)
            self.viewer.sync()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            if self.render_mode == "human":
                self.viewer.close()
            self.viewer = None
