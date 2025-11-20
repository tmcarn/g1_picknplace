import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
import time


class G1SquatEnv(gym.Env):
    def __init__(self, render_mode='human', render_fps=30, policy_feq=50):
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

        # Get dimensions
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Get actuator limits
        action_low = self.model.actuator_ctrlrange[:, 0]
        action_high = self.model.actuator_ctrlrange[:, 1]

        # Define action space
        self.action_space = Box(
            low=action_low,
            high=action_high,
            dtype=np.float64
        )

        # Squat parameters (must be defined before observation space)
        self.target_height_high = 0.79  # Standing height
        self.target_height_low = 0.50   # Squatting height
        self.squat_cycle_steps = 150    # Steps per squat cycle
        self.stand_duration = 50        # Steps to stay standing
        self.squat_duration = 50        # Steps to stay in squat
        self.transition_duration = 25   # Steps for transition between poses
        self.step_count = 0

        obs_low, obs_high = self.get_obs_limits()
        self.observation_space = Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64
        )

        # Load the standing keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.data.qpos[:] = self.model.key_qpos[keyframe_id]
        self.data.qvel[:] = self.model.key_qvel[keyframe_id]
        mujoco.mj_forward(self.model, self.data)
        print("Loaded in 'stand' keyframe as initial position")
        print(f"Action Space: {action_high.shape}")
        print(f"Observation Space: {obs_high.shape}")
        print(f"{'='*50}")

        # Initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Perturbation Noise
        self.reset_noise = 0.005
        
        # Reward Parameters
        self.min_term_height = 0.3  # Terminate if robot falls too low
        term_tilt_angle = 45
        self.term_tilt_tresh = self.calculate_tilt_thresh(term_tilt_angle)

        # Environment Constants
        self.up_vector = np.array([0, 0, 1])

        # Freeze Hand DOFs
        self.wrist_and_hand_actuators = list(range(19, 29)) + list(range(33, 43))

        # Symmetry indices
        self.left_leg_qpos_indices = [7, 8, 9, 10, 11, 12]
        self.right_leg_qpos_indices = [13, 14, 15, 16, 17, 18]
        self.left_arm_qpos_indices = [22, 23, 24, 25]
        self.right_arm_qpos_indices = [36, 37, 38, 39]

        self.left_leg_qvel_indices = [6, 7, 8, 9, 10, 11]
        self.right_leg_qvel_indices = [12, 13, 14, 15, 16, 17]
        self.left_arm_qvel_indices = [21, 22, 23, 24]
        self.right_arm_qvel_indices = [35, 36, 37, 38]
        
        # Knee joint indices (typically the 4th joint in leg sequence)
        self.left_knee_idx = 10   # left_knee_joint
        self.right_knee_idx = 16  # right_knee_joint

        # Render 
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps
        self.frame_skip = int((1 / self.model.opt.timestep) / policy_feq)

    def calculate_tilt_thresh(self, tilt_degrees):
        tilt_rad = np.radians(tilt_degrees)
        threshold = np.cos(tilt_rad)
        return threshold

    def get_target_height(self):
        """Calculate target height based on current step in squat cycle
        Phase sequence:
        1. Stand (hold high position)
        2. Transition down (bend)
        3. Squat (hold low position)
        4. Transition up (stand back up)
        """
        cycle_pos = self.step_count % self.squat_cycle_steps
        
        # Phase 1: Standing (0 to stand_duration)
        if cycle_pos < self.stand_duration:
            return self.target_height_high
        
        # Phase 2: Transition down (stand_duration to stand_duration + transition_duration)
        elif cycle_pos < self.stand_duration + self.transition_duration:
            progress = (cycle_pos - self.stand_duration) / self.transition_duration
            # Smooth transition using cosine interpolation
            smooth_progress = (1 - np.cos(progress * np.pi)) / 2
            return self.target_height_high - (self.target_height_high - self.target_height_low) * smooth_progress
        
        # Phase 3: Squatting (stand_duration + transition_duration to stand_duration + transition_duration + squat_duration)
        elif cycle_pos < self.stand_duration + self.transition_duration + self.squat_duration:
            return self.target_height_low
        
        # Phase 4: Transition up (remaining time in cycle)
        else:
            time_in_phase = cycle_pos - (self.stand_duration + self.transition_duration + self.squat_duration)
            progress = time_in_phase / self.transition_duration
            # Smooth transition using cosine interpolation
            smooth_progress = (1 - np.cos(progress * np.pi)) / 2
            return self.target_height_low + (self.target_height_high - self.target_height_low) * smooth_progress

    def get_obs(self):
        """Get observation from current state."""
        target_height = self.get_target_height()
        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            [target_height]  # Add target height so robot knows what to do
        ])
        return obs.astype(np.float64)
    
    def get_obs_limits(self):
        """Get observation space limits from model."""
        qpos_low = np.full(self.nq, -np.inf)
        qpos_high = np.full(self.nq, np.inf)
        
        # Set limits for actuated joints
        for i in range(1, self.model.njnt):
            joint_type = self.model.jnt_type[i]
            if joint_type == 3:  # Hinge joint
                qpos_addr = self.model.jnt_qposadr[i]
                qpos_low[qpos_addr] = self.model.jnt_range[i, 0]
                qpos_high[qpos_addr] = self.model.jnt_range[i, 1]
        
        # Quaternions in [-1, 1]
        qpos_low[3:7] = -1.0
        qpos_high[3:7] = 1.0
        
        # Velocity limits
        qvel_low = np.full(self.nv, -20.0)
        qvel_high = np.full(self.nv, 20.0)
        
        # Target height limits
        target_low = np.array([self.target_height_low])
        target_high = np.array([self.target_height_high])
        
        obs_low = np.concatenate([qpos_low, qvel_low, target_low]).astype(np.float64)
        obs_high = np.concatenate([qpos_high, qvel_high, target_high]).astype(np.float64)
        
        return obs_low, obs_high

    def calculate_reward(self):
        '''
        Reward for Squatting Motion:
        - Height tracking (follow target height)
        - Upright orientation
        - Smooth motion (penalize high velocities)
        - Symmetry
        - Control cost
        '''
        # Current height
        height = self.data.qpos[2]
        target_height = self.get_target_height()
        
        # Height tracking reward - heavily weight this so robot learns to follow target
        height_error = abs(height - target_height)
        height_reward = 20.0 * np.exp(-10.0 * height_error)  # Increased weight and steepness
        
        # Knee angle reward - encourage bending when target is low
        left_knee = self.data.qpos[self.left_knee_idx]
        right_knee = self.data.qpos[self.right_knee_idx]
        avg_knee = (left_knee + right_knee) / 2
        
        # Map target height to desired knee angle (negative = bent)
        # Standing (0.79m) -> knee ~0, Squatting (0.50m) -> knee ~-1.5 rad
        height_range = self.target_height_high - self.target_height_low
        target_knee = -1.5 * (self.target_height_high - target_height) / height_range
        knee_error = abs(avg_knee - target_knee)
        knee_reward = 5.0 * np.exp(-5.0 * knee_error)

        # Upright orientation reward
        upright_reward = 3.0 * self.get_tilt()

        # Smooth motion - penalize high velocities
        velocity_penalty = -0.01 * np.sum(np.square(self.data.qvel))

        # Symmetry reward
        left_leg_qpos = self.data.qpos[self.left_leg_qpos_indices]
        right_leg_qpos = self.data.qpos[self.right_leg_qpos_indices]
        leg_pos_diff = np.sum(np.square(left_leg_qpos - right_leg_qpos))

        left_leg_qvel = self.data.qvel[self.left_leg_qvel_indices]
        right_leg_qvel = self.data.qvel[self.right_leg_qvel_indices]
        leg_vel_diff = np.sum(np.square(left_leg_qvel - right_leg_qvel))

        symmetry_reward = -0.1 * (leg_pos_diff + leg_vel_diff)

        # Control cost
        ctrl_cost = -0.001 * np.sum(np.square(self.data.ctrl))

        # Bonus for staying alive
        alive_bonus = 1.0

        total_reward = (height_reward + knee_reward + upright_reward + velocity_penalty + 
                        symmetry_reward + ctrl_cost + alive_bonus)
        
        return total_reward
    
    def get_tilt(self):
        torso_quat = self.data.qpos[3:7]
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, torso_quat)
        torso_z_axis = rotation_matrix[6:9]
        return np.dot(torso_z_axis, self.up_vector)

    def terminate(self):
        height = self.data.qpos[2]
        if height < self.min_term_height:
            return True
        
        if self.get_tilt() < self.term_tilt_tresh:
            return True
        
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
        
        # Reset to initial state with small random perturbations
        self.data.qpos[:] = self.init_qpos + np.random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nq
        )
        self.data.qvel[:] = self.init_qvel + np.random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nv
        )
        
        # Reset step counter
        self.step_count = 0
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = self.get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Take a step in the environment"""
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action
        self.data.ctrl[:] = action
        # Freeze hand and wrist DOFs
        self.data.ctrl[self.wrist_and_hand_actuators] = 0
        
        # Step through physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Get observation
        obs = self.get_obs()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check termination
        terminate = self.terminate()
        truncated = False
        
        info = {
            'height': self.data.qpos[2],
            'target_height': self.get_target_height(),
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
