import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
import time
from dataclasses import dataclass
from typing import List


@dataclass
class G1SquatConfig:
    """Configuration for G1 humanoid squatting task"""
    # Task parameters
    target_squat_depth: float = 0.29  # meters below standing height (0.79 - 0.50 = 0.29m)
    squat_hold_time: float = 1.0      # seconds to hold squat position
    stand_hold_time: float = 1.0      # seconds to hold standing position
    
    # Phase thresholds
    phase_transition_threshold: float = 0.05  # meters tolerance for phase transitions
    
    # Reward weights
    height_tracking_weight: float = 15.0
    stability_weight: float = 3.0
    energy_penalty_weight: float = 0.001
    upright_bonus_weight: float = 3.0
    smooth_motion_weight: float = 1.0
    symmetry_weight: float = 1.0
    knee_tracking_weight: float = 5.0
    
    # Safety parameters
    min_term_height: float = 0.3
    term_tilt_angle: float = 45  # degrees
    
    # Training parameters
    action_scale: float = 1.0
    reset_noise: float = 0.005


class G1SquatEnv(gym.Env):
    def __init__(self, render_mode='human', render_fps=30, policy_feq=50, config: G1SquatConfig = None):
        super().__init__()
        
        xml_path = "unitree_g1/g1_mocap_29dof_with_hands.xml"
        
        # Select a graphics backend for the viewer
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Configuration
        self.config = config if config is not None else G1SquatConfig()

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

        # Phase state machine
        self.phase = "standing"  # standing, descending, holding, ascending
        self.phase_timer = 0.0
        self.squat_count = 0
        self.initial_height = 0.79  # G1 standing height
        
        # Target joint positions for squatting
        self.squat_joint_targets = self._get_squat_targets()

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
        self.reset_noise = self.config.reset_noise
        
        # Reward Parameters
        self.min_term_height = self.config.min_term_height
        self.term_tilt_thresh = np.cos(np.radians(self.config.term_tilt_angle))

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
        
        # Knee joint indices
        self.left_knee_idx = 10   # left_knee_joint
        self.right_knee_idx = 16  # right_knee_joint
        
        # Previous action for smoothness reward
        self.previous_action = None

        # Render 
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps
        self.frame_skip = int((1 / self.model.opt.timestep) / policy_feq)

    def _get_squat_targets(self):
        """Define target knee angles for squatting"""
        return {
            'standing_knee': 0.0,
            'squat_knee': -1.5,  # Bent knees for squat
        }

    def _get_target_height(self):
        """Get target height based on current phase"""
        if self.phase in ["descending", "holding"]:
            return self.initial_height - self.config.target_squat_depth
        else:  # standing or ascending
            return self.initial_height
    
    def _get_target_knee_angle(self):
        """Get target knee angle based on current phase"""
        if self.phase in ["descending", "holding"]:
            return self.squat_joint_targets['squat_knee']
        else:
            return self.squat_joint_targets['standing_knee']

    def _get_phase_encoding(self):
        """One-hot encode current phase"""
        phase_map = {"standing": 0, "descending": 1, "holding": 2, "ascending": 3}
        encoding = np.zeros(4)
        encoding[phase_map[self.phase]] = 1.0
        return encoding

    def get_obs(self):
        """Get observation with phase information"""
        target_height = self._get_target_height()
        target_knee = self._get_target_knee_angle()
        phase_encoding = self._get_phase_encoding()
        timer_normalized = min(self.phase_timer / 2.0, 1.0)
        
        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            [target_height],
            [target_knee],
            phase_encoding,
            [timer_normalized],
            [self.squat_count / 10.0]
        ])
        return obs.astype(np.float64)
    
    def get_obs_limits(self):
        """Get observation space limits"""
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
        target_height_low = np.array([self.initial_height - self.config.target_squat_depth])
        target_height_high = np.array([self.initial_height])
        
        # Target knee limits
        target_knee_low = np.array([self.squat_joint_targets['squat_knee']])
        target_knee_high = np.array([self.squat_joint_targets['standing_knee']])
        
        # Phase encoding (4), timer (1), squat count (1)
        extra_low = np.array([0, 0, 0, 0, 0, 0])
        extra_high = np.array([1, 1, 1, 1, 1, 1])
        
        obs_low = np.concatenate([qpos_low, qvel_low, target_height_low, target_knee_low, extra_low]).astype(np.float64)
        obs_high = np.concatenate([qpos_high, qvel_high, target_height_high, target_knee_high, extra_high]).astype(np.float64)
        
        return obs_low, obs_high

    def calculate_reward(self):
        """Phase-based reward calculation"""
        reward = 0.0
        
        # Current state
        height = self.data.qpos[2]
        target_height = self._get_target_height()
        left_knee = self.data.qpos[self.left_knee_idx]
        right_knee = self.data.qpos[self.right_knee_idx]
        avg_knee = (left_knee + right_knee) / 2
        target_knee = self._get_target_knee_angle()
        
        # Phase-specific rewards
        if self.phase == "standing":
            # Reward for stable standing
            height_error = abs(height - self.initial_height)
            stand_reward = np.exp(-20.0 * height_error)
            reward += self.config.height_tracking_weight * stand_reward
            
            # Bonus for maintaining upright stance
            vel_magnitude = np.linalg.norm(self.data.qvel[:6])
            stability = np.exp(-2.0 * vel_magnitude)
            reward += self.config.stability_weight * stability
            
        elif self.phase == "descending":
            # Reward progress toward squat position
            height_progress = (self.initial_height - height) / self.config.target_squat_depth
            height_progress = np.clip(height_progress, 0, 1)
            reward += self.config.height_tracking_weight * height_progress
            
            # Knee angle tracking
            knee_progress = (self.squat_joint_targets['standing_knee'] - avg_knee) / abs(self.squat_joint_targets['squat_knee'])
            knee_progress = np.clip(knee_progress, 0, 1)
            reward += self.config.knee_tracking_weight * knee_progress
            
            # Bonus for reaching target
            if abs(height - target_height) < self.config.phase_transition_threshold:
                reward += 10.0
                
        elif self.phase == "holding":
            # Reward for maintaining squat position
            height_error = abs(height - target_height)
            hold_reward = np.exp(-20.0 * height_error)
            reward += self.config.height_tracking_weight * hold_reward
            
            # Stability during hold
            vel_magnitude = np.linalg.norm(self.data.qvel[:6])
            stability = np.exp(-3.0 * vel_magnitude)
            reward += self.config.stability_weight * stability * 2.0  # Double weight during hold
            
            # Knee maintenance
            knee_error = abs(avg_knee - target_knee)
            knee_hold = np.exp(-5.0 * knee_error)
            reward += self.config.knee_tracking_weight * knee_hold
            
        elif self.phase == "ascending":
            # Reward progress back to standing
            height_progress = (height - target_height) / self.config.target_squat_depth
            height_progress = np.clip(height_progress, 0, 1)
            reward += self.config.height_tracking_weight * height_progress
            
            # Knee angle tracking
            knee_progress = (avg_knee - self.squat_joint_targets['squat_knee']) / abs(self.squat_joint_targets['squat_knee'])
            knee_progress = np.clip(knee_progress, 0, 1)
            reward += self.config.knee_tracking_weight * knee_progress
            
            # Bonus for reaching standing
            if abs(height - self.initial_height) < self.config.phase_transition_threshold:
                reward += 10.0
        
        # Always-active rewards
        
        # Upright orientation
        upright_reward = self.config.upright_bonus_weight * self.get_tilt()
        reward += upright_reward
        
        # Symmetry (legs should move together)
        left_leg_qpos = self.data.qpos[self.left_leg_qpos_indices]
        right_leg_qpos = self.data.qpos[self.right_leg_qpos_indices]
        leg_symmetry = np.exp(-np.linalg.norm(left_leg_qpos - right_leg_qpos))
        reward += self.config.symmetry_weight * leg_symmetry
        
        # Smooth motion
        if self.previous_action is not None:
            action_diff = np.linalg.norm(self.data.ctrl - self.previous_action)
            smooth_reward = np.exp(-2.0 * action_diff)
            reward += self.config.smooth_motion_weight * smooth_reward
        
        # Energy penalty
        energy = np.sum(np.square(self.data.ctrl))
        reward -= self.config.energy_penalty_weight * energy
        
        # Alive bonus
        reward += 1.0
        
        return reward
    
    def _update_phase(self):
        """Update phase based on current state and timer"""
        height = self.data.qpos[2]
        dt = self.model.opt.timestep * self.frame_skip
        self.phase_timer += dt
        
        if self.phase == "standing":
            if self.phase_timer >= self.config.stand_hold_time:
                self.phase = "descending"
                self.phase_timer = 0.0
                
        elif self.phase == "descending":
            target = self.initial_height - self.config.target_squat_depth
            if abs(height - target) < self.config.phase_transition_threshold:
                self.phase = "holding"
                self.phase_timer = 0.0
                
        elif self.phase == "holding":
            if self.phase_timer >= self.config.squat_hold_time:
                self.phase = "ascending"
                self.phase_timer = 0.0
                
        elif self.phase == "ascending":
            if abs(height - self.initial_height) < self.config.phase_transition_threshold:
                self.phase = "standing"
                self.phase_timer = 0.0
                self.squat_count += 1
    
    def get_tilt(self):
        torso_quat = self.data.qpos[3:7]
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, torso_quat)
        torso_z_axis = rotation_matrix[6:9]
        return np.dot(torso_z_axis, self.up_vector)

    def terminate(self):
        height = self.data.qpos[2]
        tilt = self.get_tilt()
        
        if height < self.min_term_height:
            return True
        if tilt < self.term_tilt_thresh:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial state with noise
        noise = self.reset_noise * np.random.randn(self.nq)
        noise[3:7] = 0  # Don't perturb quaternion
        self.data.qpos[:] = self.init_qpos + noise
        self.data.qvel[:] = self.init_qvel + self.reset_noise * np.random.randn(self.nv)
        
        # Reset phase
        self.phase = "standing"
        self.phase_timer = 0.0
        self.squat_count = 0
        self.previous_action = None
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = self.get_obs()
        info = {'phase': self.phase, 'squat_count': self.squat_count}
        
        return obs, info

    def step(self, action):
        # Freeze hand actuators
        action[self.wrist_and_hand_actuators] = 0
        
        # Store for smoothness reward
        self.previous_action = action.copy()
        
        # Apply action
        self.data.ctrl[:] = action
        
        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Update phase
        self._update_phase()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check termination
        terminated = self.terminate()
        truncated = False
        
        # Get observation
        obs = self.get_obs()
        
        # Info
        info = {
            'phase': self.phase,
            'squat_count': self.squat_count,
            'phase_timer': self.phase_timer,
            'target_height': self._get_target_height(),
            'current_height': self.data.qpos[2]
        }
        
        # Render
        if self.render_mode == 'human':
            self.render()
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
