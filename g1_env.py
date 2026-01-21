import os
import numpy as np
import mujoco
import mujoco.viewer

from scipy.spatial.transform import Rotation as R


class MJG1Env:
    '''
    Controls the Actions of G1 and the resulting States 
    '''

    def __init__(self, render_mode="human", render_fps=30):
        self.xml_path = "unitree_g1/g1_with_box_torque_ctrl.xml"
        
        # Select a graphics backend for the viewer
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Load the standing keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.data.qpos[:] = self.model.key_qpos[keyframe_id]
        self.data.qvel[:] = self.model.key_qvel[keyframe_id]

        self.data.qvel[:] = 0 # Sets all initial joint velocities to zero

        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        mujoco.mj_forward(self.model, self.data) # Update Step

        print("="*50)
        print("Successfully loaded in 'stand' keyframe as initial position")
        print("="*50)

        self.print_model_info()

        # Environmental Constants
        self.g = 9.81
        self.grav_vec = np.array([0,0,-9.81])

        _, self.box_mass = self.get_obj_com()
        _, self.g1_mass = self.get_g1_com()
        self.total_mass = self.g1_mass + self.box_mass

        # Initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Set Desired state to be equal to Initial State
        self.x_desired = self.get_state()

        # Constraints Parameters
        # TODO: Set these limits based on something more concrete
        self.mu = 0.7
        self.F_min = self.total_mass * self.g * 0.05   # Minimum normal force (N) - keeps contact
        self.F_max = self.total_mass * self.g * 10  # Maximum normal force (N) - robot/ground limits
        self.M_max = 10.0   # Maximum moment (N⋅m)

        # RENDER CONFIG
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps
             
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
            return robot_com, total_mass

    def get_obj_com(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        obj_com = self.data.xpos[box_id]
        obj_mass = self.model.body_mass[box_id]
        return obj_com, obj_mass
    
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
    
    def get_orientation_jacobian(self):
        """
        Get base orientation Jacobian
        Maps: q̇ → ω (angular velocity)
        
        Returns:
            J_orient: (3, nv) Jacobian
        """
        nv = self.model.nv
        jac_orient = np.zeros((3, nv))
        
        # For floating base, angular velocity is DOFs 3:6
        jac_orient[:, 3:6] = np.eye(3)
        
        return jac_orient

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