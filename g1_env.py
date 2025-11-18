import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
import time


class G1Env(gym.Env):
    def __init__(self, render_mode='human', render_fps=30, policy_feq=50, ): # 50 Hz 
        super().__init__()
        
        xml_path = "g1_picknplace_controller/unitree_g1/g1_mocap_29dof_with_hands.xml"
        
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
        self.reset_noise = 0.01

        # Reward Parameters
        self.min_reward_height = 0.75 # meters
        self.min_term_height = 0.5
        
        term_tilt_angle = 30 # degrees
        self.term_tilt_tresh = self.calculate_tilt_thresh(term_tilt_angle)

        # Environment Constants
        self.up_vector = np.array([0, 0, 1])
        self.gravity = np.array([0,0,-9.81])
        self.box_mass = 10 #kg

        # Render 
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps

    
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
        
        # Torso position: no strict limits (can move anywhere)
        pos_low = np.full(3, -np.inf)
        pos_high = np.full(3, np.inf)
        
        # Quaternion: normalized, so each component in [-1, 1]
        quat_low = np.full(4, -1.0)
        quat_high = np.full(4, 1.0)
        
        # Joint positions: skip free joint (index 0)
        joint_pos_low = self.model.jnt_range[1:, 0]  # Shape: (n_joints-1,)
        joint_pos_high = self.model.jnt_range[1:, 1]
        
        # Velocities - need to match the structure of get_obs()
        # Torso linear velocity (3)
        vel_low = np.full(3, -10.0)  # m/s
        vel_high = np.full(3, 10.0)
        
        # Torso angular velocity (3)
        ang_vel_low = np.full(3, -10.0)  # rad/s
        ang_vel_high = np.full(3, 10.0)
        
        # Joint velocities: skip free joint DOFs (first 6)
        # Free joint has 6 velocity DOFs: 3 linear + 3 angular
        joint_vel_low = np.full(self.nv - 6, -20.0)  # rad/s
        joint_vel_high = np.full(self.nv - 6, 20.0)
        
        # Concatenate all limits - MUST match order in get_obs()
        obs_low = np.concatenate([
            pos_low,           # 3
            quat_low,          # 4
            joint_pos_low,     # nq - 7
            vel_low,           # 3
            ang_vel_low,       # 3
            joint_vel_low,     # nv - 6
        ]).astype(np.float64)
        
        obs_high = np.concatenate([
            pos_high,          # 3
            quat_high,         # 4
            joint_pos_high,    # nq - 7
            vel_high,          # 3
            ang_vel_high,      # 3
            joint_vel_high,    # nv - 6
        ]).astype(np.float64)
        
        return obs_low, obs_high

    def calculate_reward(self):
        '''
        Simple Reward Policy for Humanoid Ballancing
        Reward:
        - Torso Height (Z position)
        - Upright Torso Orientation (distance beween normal vector and torso z axis)

        Cost:
        - Penalize Expensive Actions
        '''
        # Torso height (penalize falling)
        height = self.data.qpos[2]
        height_reward = 1.5 if height > self.min_reward_height else 0.0

        upright_reward = 1.5 * self.get_tilt()

        # Control Cost (Penalizes Large Actions)
        ctrl_cost = -1 * np.sum(np.square(self.data.ctrl))

        total_reward = height_reward + upright_reward + ctrl_cost
        
        return total_reward
    
    def get_tilt(self):
        # Upright orientation
        torso_quat = self.data.qpos[3:7]
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, torso_quat)
        torso_z_axis = rotation_matrix[6:9]
        return np.dot(torso_z_axis, self.up_vector)
    
    def apply_force(self):
        box_force = self.gravity * self.box_mass # F = ma
        force_per_hand = box_force / 2

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
