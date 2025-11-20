import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
import time


class G1Env(gym.Env):
    def __init__(self, render_mode='human', render_fps=30, policy_feq=50, external_load_kg: float = 0.0): # 50 Hz 
        super().__init__()
        
        xml_path = "unitree_g1/g1_mocap_29dof_with_hands.xml"
        
        # Select a graphics backend for the viewer
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Print model info
        print(f"\n{'='*50}")
        print(f"Model: {xml_path}")
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
        self.frame_skip = int((1 / self.model.opt.timestep) / policy_feq)
        print(f"Frame skip: {self.frame_skip}")
        print(f"Control frequency: {1/(self.model.opt.timestep * self.frame_skip)} Hz")

        # Get dimensions
        self.nu = self.model.nu  # Number of actuators
        self.nq = self.model.nq  # Number of position coordinates
        self.nv = self.model.nv  # Number of velocity coordinates

        # Get actuator limits
        action_low = self.model.actuator_ctrlrange[:, 0]
        action_high = self.model.actuator_ctrlrange[:, 1]

        # Define action space (position targets)
        self.action_space = Box(
            low=action_low,
            high=action_high,
            dtype=np.float64
        )
        print(f"Action Space:{action_high.shape}")

        obs_low, obs_high = self.get_obs_limits()

        self.observation_space = Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64
        )

        print(f"Observation Space:{obs_high.shape}")

        # Load the standing keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.data.qpos[:] = self.model.key_qpos[keyframe_id]
        self.data.qvel[:] = self.model.key_qvel[keyframe_id]
        mujoco.mj_forward(self.model, self.data)
        print("Loaded in 'stand' keyframe as initial position")
    

        # Initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Perturbation Noise
        self.reset_noise = 0.005  # Reduced noise for easier learning

        # Reward Parameters
        self.min_reward_height = 0.70 # meters (slightly lower threshold)
        self.min_term_height = 0.4  # Lower termination threshold
        
        term_tilt_angle = 45 # degrees (more tolerant)
        self.term_tilt_tresh = self.calculate_tilt_thresh(term_tilt_angle)

        # Environment Constants
        self.up_vector = np.array([0, 0, 1])
        self.gravity = np.array([0,0,-9.81])
        self.box_mass = 10 #kg
        self.external_load_mass = max(external_load_kg, 0.0)

        # Render 
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

        # Symmetry Reward - using qpos indices (skip floating base at 0-6)
        # Left leg joints in qpos: 7-12
        # Right leg joints in qpos: 13-18
        # Left arm joints in qpos: 22-25 (shoulder pitch/roll/yaw, elbow)
        # Right arm joints in qpos: 36-39 (shoulder pitch/roll/yaw, elbow)
        self.left_leg_qpos_indices = [7, 8, 9, 10, 11, 12]
        self.right_leg_qpos_indices = [13, 14, 15, 16, 17, 18]
        self.left_arm_qpos_indices = [22, 23, 24, 25]
        self.right_arm_qpos_indices = [36, 37, 38, 39]

        # Velocity indices are offset by 6 for the floating base
        # Left leg in qvel: 6-11
        # Right leg in qvel: 12-17
        # Left arm in qvel: 21-24
        # Right arm in qvel: 35-38
        self.left_leg_qvel_indices = [6, 7, 8, 9, 10, 11]
        self.right_leg_qvel_indices = [12, 13, 14, 15, 16, 17]
        self.left_arm_qvel_indices = [21, 22, 23, 24]
        self.right_arm_qvel_indices = [35, 36, 37, 38]

        # Freeze Hand DOFs
        self.wrist_and_hand_actuators = list(range(19, 29)) + list(range(33, 43))


    
    def calculate_tilt_thresh(self, tilt_degrees):
        tilt_rad = np.radians(tilt_degrees)
        threshold = np.cos(tilt_rad)
        return threshold

    def get_obs(self):
        """Get observation from current state."""
        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])
        return obs.astype(np.float64)
    
    def get_obs_limits(self):
        """Get observation space limits from model."""
        
        # Position limits for all qpos (including robot + box)
        # For free joints (robot and box): position is unbounded, quaternion is normalized
        qpos_low = np.full(self.nq, -np.inf)
        qpos_high = np.full(self.nq, np.inf)
        
        # Set limits for actuated joints (indices 7 to 49 for robot joints, excluding floating base at 0-6 and box at 50-56)
        for i in range(1, self.model.njnt):  # Skip first freejoint (robot base)
            joint_type = self.model.jnt_type[i]
            if joint_type == 3:  # Hinge joint
                qpos_addr = self.model.jnt_qposadr[i]
                qpos_low[qpos_addr] = self.model.jnt_range[i, 0]
                qpos_high[qpos_addr] = self.model.jnt_range[i, 1]
        
        # Quaternions should be in [-1, 1] for both robot and box
        qpos_low[3:7] = -1.0  # Robot quaternion
        qpos_high[3:7] = 1.0
        if self.nq >= 54:  # If box exists
            qpos_low[50:54] = -1.0  # Box quaternion
            qpos_high[50:54] = 1.0
        
        # Velocity limits for all qvel (including robot + box)
        qvel_low = np.full(self.nv, -20.0)
        qvel_high = np.full(self.nv, 20.0)
        
        # Concatenate all limits - MUST match order in get_obs()
        obs_low = np.concatenate([
            qpos_low,
            qvel_low,
        ]).astype(np.float64)
        
        obs_high = np.concatenate([
            qpos_high,
            qvel_high,
        ]).astype(np.float64)
        
        return obs_low, obs_high

    def calculate_reward(self):
        '''
        Improved Reward Policy for Humanoid Standing
        Reward:
        - Torso Height (Z position) - increased weight
        - Upright Torso Orientation (distance between normal vector and torso z axis) - increased weight
        - Stability (penalize high velocities)
        - Symmetry Reward
        - Motion Reference Reward
        - Support Polygon Reward

        Cost:
        - Penalize Expensive Actions (but less harsh)
        '''
        # Torso height reward (stronger signal)
        height = self.data.qpos[2]
        # Continuous height reward instead of binary
        target_height = 0.95  # Target standing height
        height_reward = 5.0 * max(0, 1.0 - abs(height - target_height))

        # Upright orientation reward (stronger signal)
        upright_reward = 5.0 * self.get_tilt()

        # Stability reward - penalize high velocities
        velocity_penalty = -0.01 * np.sum(np.square(self.data.qvel))

        # Control Cost (less harsh, only penalize extreme actions)
        ctrl_cost = -0.001 * np.sum(np.square(self.data.ctrl))

        # --- Symmetry Reward ---
        # Arms
        left_arm_qpos = self.data.qpos[self.left_arm_qpos_indices]
        right_arm_qpos = self.data.qpos[self.right_arm_qpos_indices]
        arm_pos_diff = np.sum(np.square(left_arm_qpos - right_arm_qpos))

        left_arm_qvel = self.data.qvel[self.left_arm_qvel_indices]
        right_arm_qvel = self.data.qvel[self.right_arm_qvel_indices]
        arm_vel_diff = np.sum(np.square(left_arm_qvel - right_arm_qvel))

        # Legs
        left_leg_qpos = self.data.qpos[self.left_leg_qpos_indices]
        right_leg_qpos = self.data.qpos[self.right_leg_qpos_indices]
        leg_pos_diff = np.sum(np.square(left_leg_qpos - right_leg_qpos))

        left_leg_qvel = self.data.qvel[self.left_leg_qvel_indices]
        right_leg_qvel = self.data.qvel[self.right_leg_qvel_indices]
        leg_vel_diff = np.sum(np.square(left_leg_qvel - right_leg_qvel))

        symmetry_reward = -0.1 * (arm_pos_diff + arm_vel_diff + leg_pos_diff + leg_vel_diff)

        # --- Motion Reference Reward ---
        qpos_deviation = np.sum(np.square(self.data.qpos - self.init_qpos))
        motion_ref_reward = -0.2 * qpos_deviation

        # --- Support Polygon Reward ---
        com_pos = self.data.subtree_com[0, :2]
        left_foot_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'left_foot')][:2]
        right_foot_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'right_foot')][:2]
        
        foot_center = (left_foot_pos + right_foot_pos) / 2.0
        com_dist_from_center = np.linalg.norm(com_pos - foot_center)
        support_polygon_reward = -0.5 * com_dist_from_center

        # Bonus for staying alive
        alive_bonus = 1.0

        total_reward = (height_reward + upright_reward + velocity_penalty + ctrl_cost + 
                        symmetry_reward + motion_ref_reward + support_polygon_reward + alive_bonus)
        
        return total_reward
    
    def get_tilt(self):
        # Upright orientation
        torso_quat = self.data.qpos[3:7]
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, torso_quat)
        torso_z_axis = rotation_matrix[6:9]
        return np.dot(torso_z_axis, self.up_vector)
    
    def apply_force(self):
        if self.external_load_mass <= 0:
            return
            
        load_force = self.gravity * self.external_load_mass # F = ma (vector)
        force_per_hand = load_force / 2

        hand_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        for hand_name in hand_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
            
            # Apply downward force (negative z)

            # Get hand position in world frame
            hand_pos = self.data.xpos[body_id].copy()
            
            # Apply force in world frame at hand position
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_per_hand,      # Force in WORLD frame
                np.zeros(3),         # No torque
                hand_pos,            # Point of application (world coords)
                body_id,             # Body ID
                self.data.qfrc_applied  # Output array
            )
            

    def terminate(self):
        # Torso height (penalize falling)
        height = self.data.qpos[2]
        if height < self.min_term_height:
            return True
        
        if self.get_tilt() < self.term_tilt_tresh:
            return True
        
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

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
        
        obs = self.get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Take a step in the environment"""

        # Clear forces from previous step
        self.data.qfrc_applied[:] = 0

        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action (as control signal)
        self.data.ctrl[:] = action
        # Freeze hand and wrist DOFs
        self.data.ctrl[self.wrist_and_hand_actuators] = 0

        # Apply external load if configured
        if self.external_load_mass > 0:
            self.apply_force()

        # Step through physics multiple times with same action (frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self.get_obs()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check termination conditions
        terminate = self.terminate()
        truncated = False
        
        info = {
            'height': self.data.qpos[2],
            'reward': reward,
        }
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminate, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Sleep to match desired render FPS
            time.sleep(self.render_dt)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model, self._render_height, self._render_width)
            self.viewer.update_scene(self.data)
            return self.viewer.render()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            if self.render_mode == "human":
                self.viewer.close()
            self.viewer = None


# g1 = G1Env()
