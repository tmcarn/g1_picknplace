"""
Hybrid MPC + RL Controller for G1 Standing
Uses MPC for baseline stability and RL to learn residual corrections
"""
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
import time


class G1HybridEnv(gym.Env):
    """Environment that uses PD controller as baseline + RL learns corrections"""
    
    def __init__(self, render_mode='human', render_fps=30, policy_freq=50):
        super().__init__()
        
        xml_path = "unitree_g1/g1_mocap_29dof_with_hands.xml"
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Load standing pose as target
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.target_qpos = self.model.key_qpos[keyframe_id].copy()
        self.target_qvel = self.model.key_qvel[keyframe_id].copy()
        
        # PD controller gains
        self.kp = 100.0
        self.kd = 10.0
        
        # Physics parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = int(1.0 / (policy_freq * self.dt))
        
        # Action space: RL learns CORRECTIONS to PD controller output
        # Much smaller action space = faster learning!
        correction_limit = 20.0  # Max correction torque
        self.action_space = Box(
            low=-correction_limit,
            high=correction_limit,
            shape=(self.model.nu,),
            dtype=np.float64
        )
        
        # Observation space (same as before)
        obs_low, obs_high = self.get_obs_limits()
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float64)
        
        # Load initial state
        self.init_qpos = self.target_qpos.copy()
        self.init_qvel = self.target_qvel.copy()
        self.reset_noise = 0.005
        
        # Reward parameters
        self.min_reward_height = 0.70
        self.min_term_height = 0.4
        self.term_tilt_thresh = np.cos(np.radians(45))
        self.up_vector = np.array([0, 0, 1])
        self.gravity = np.array([0, 0, -9.81])
        self.box_mass = 10.0
        
        # Rendering
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / render_fps
        
        print("="*60)
        print("Hybrid MPC + RL Controller")
        print("="*60)
        print("Base Controller: PD (kp=100, kd=10)")
        print("RL learns: Correction torques (±20 Nm)")
        print(f"Action space: {self.action_space.shape} (smaller = faster learning!)")
        print("="*60)
    
    def get_obs_limits(self):
        """Same as original environment"""
        pos_low = np.full(3, -np.inf)
        pos_high = np.full(3, np.inf)
        quat_low = np.full(4, -1.0)
        quat_high = np.full(4, 1.0)
        joint_pos_low = self.model.jnt_range[1:, 0]
        joint_pos_high = self.model.jnt_range[1:, 1]
        vel_low = np.full(3, -10.0)
        vel_high = np.full(3, 10.0)
        ang_vel_low = np.full(3, -10.0)
        ang_vel_high = np.full(3, 10.0)
        joint_vel_low = np.full(self.model.nv - 6, -20.0)
        joint_vel_high = np.full(self.model.nv - 6, 20.0)
        
        obs_low = np.concatenate([
            pos_low, quat_low, joint_pos_low,
            vel_low, ang_vel_low, joint_vel_low
        ])
        obs_high = np.concatenate([
            pos_high, quat_high, joint_pos_high,
            vel_high, ang_vel_high, joint_vel_high
        ])
        
        return obs_low, obs_high
    
    def get_obs(self):
        """Same as original environment"""
        return np.concatenate([
            self.data.qpos[:7],      # Floating base pos + quat
            self.data.qpos[7:],      # Joint positions
            self.data.qvel[:3],      # Linear velocity
            self.data.qvel[3:6],     # Angular velocity
            self.data.qvel[6:]       # Joint velocities
        ]).astype(np.float64)
    
    def pd_controller(self):
        """Baseline PD controller that maintains standing pose"""
        qpos_error = self.target_qpos[7:] - self.data.qpos[7:]
        qvel_error = self.target_qvel[6:] - self.data.qvel[6:]
        return self.kp * qpos_error + self.kd * qvel_error
    
    def calculate_reward(self):
        """Improved reward function"""
        height = self.data.qpos[2]
        target_height = 0.95
        height_reward = 5.0 * max(0, 1.0 - abs(height - target_height))
        
        upright_reward = 5.0 * self.get_tilt()
        velocity_penalty = -0.01 * np.sum(np.square(self.data.qvel))
        
        # Penalize RL corrections (encourage minimal intervention)
        correction_penalty = -0.0001 * np.sum(np.square(self.data.ctrl - self.pd_controller()))
        
        alive_bonus = 1.0
        
        return height_reward + upright_reward + velocity_penalty + correction_penalty + alive_bonus
    
    def get_tilt(self):
        """Calculate upright orientation"""
        torso_quat = self.data.qpos[3:7]
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, torso_quat)
        torso_z_axis = rotation_matrix[6:9]
        return np.dot(torso_z_axis, self.up_vector)
    
    def apply_force(self):
        """Apply external force from holding box"""
        box_force = self.gravity * self.box_mass
        force_per_hand = box_force / 2
        
        hand_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.data.qfrc_applied[:] = 0
        
        for hand_name in hand_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
            hand_pos = self.data.xpos[body_id].copy()
            mujoco.mj_applyFT(
                self.model, self.data,
                force_per_hand, np.zeros(3),
                hand_pos, body_id,
                self.data.qfrc_applied
            )
    
    def terminate(self):
        """Check termination conditions"""
        height = self.data.qpos[2]
        if height < self.min_term_height:
            return True
        if self.get_tilt() < self.term_tilt_thresh:
            return True
        return False
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data.qpos[:] = self.init_qpos + np.random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nq
        )
        self.data.qvel[:] = self.init_qvel + np.random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nv
        )
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs(), {}
    
    def step(self, rl_correction):
        """
        Step function: PD controller + RL correction
        rl_correction: Small adjustments learned by RL
        """
        # Clear forces
        self.data.qfrc_applied[:] = 0
        
        # Clip RL corrections
        rl_correction = np.clip(rl_correction, 
                                self.action_space.low, 
                                self.action_space.high)
        
        # Compute baseline PD control
        pd_control = self.pd_controller()
        
        # Final control = PD baseline + RL correction
        self.data.ctrl[:] = pd_control + rl_correction
        
        # Apply external forces
        self.apply_force()
        
        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        obs = self.get_obs()
        reward = self.calculate_reward()
        terminate = self.terminate()
        
        info = {
            'height': self.data.qpos[2],
            'reward': reward,
            'pd_control_norm': np.linalg.norm(pd_control),
            'rl_correction_norm': np.linalg.norm(rl_correction),
        }
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminate, False, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            time.sleep(self.render_dt)
            self.viewer.sync()
    
    def close(self):
        """Close the environment"""
        if self.viewer is not None:
            self.viewer.close()


if __name__ == "__main__":
    # Test the hybrid controller
    print("\nTesting Hybrid MPC + RL Environment")
    print("This uses PD controller as baseline")
    print("RL will learn small corrections on top\n")
    
    env = G1HybridEnv(render_mode='human')
    obs, info = env.reset()
    
    print("Running with random RL corrections (zero mean)...")
    print("In training, RL will learn optimal corrections\n")
    
    for step in range(1000):
        # Random small corrections (in real training, PPO will learn these)
        rl_correction = np.random.randn(env.action_space.shape[0]) * 5.0
        
        obs, reward, done, truncated, info = env.step(rl_correction)
        
        if step % 100 == 0:
            print(f"Step {step:4d} | Height: {info['height']:.3f}m | "
                  f"Reward: {reward:7.2f} | "
                  f"PD: {info['pd_control_norm']:.1f} | "
                  f"RL: {info['rl_correction_norm']:.1f}")
        
        if done:
            print(f"\nEpisode ended at step {step}")
            obs, info = env.reset()
    
    env.close()
    print("\nTest complete!")
