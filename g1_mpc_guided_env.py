import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer
import os
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MPCConfig:
    """Configuration for MPC controller"""
    horizon: int = 20  # MPC prediction horizon
    dt: float = 0.02  # MPC timestep (50 Hz)
    
    # Cost weights
    state_weight: float = 1.0
    control_weight: float = 0.01
    reference_weight: float = 10.0
    
    # Target pose
    target_height: float = 0.79
    target_com_x: float = 0.0
    target_com_y: float = 0.0


class SimpleMPC:
    """Simple MPC controller for generating reference trajectories"""
    
    def __init__(self, model: mujoco.MjModel, config: MPCConfig):
        self.model = model
        self.config = config
        self.nq = model.nq
        self.nv = model.nv
        self.nu = model.nu
        
    def compute_reference_action(self, data: mujoco.MjData) -> np.ndarray:
        """
        Compute reference action using simple MPC heuristics.
        In a full implementation, this would solve an optimization problem.
        
        For now, we use a PD controller toward the standing pose.
        """
        # Current state
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        # Get standing reference from keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        qpos_ref = self.model.key_qpos[keyframe_id]
        qvel_ref = np.zeros(self.nv)
        
        # Simple PD control
        kp = 100.0  # Position gain
        kd = 10.0   # Velocity gain
        
        # Compute PD action for each actuated joint
        action = np.zeros(self.nu)
        
        # Skip floating base (first 7 qpos elements)
        for i in range(self.nu):
            # Map actuator to joint
            joint_id = self.model.actuator_trnid[i, 0]
            joint_type = self.model.jnt_type[joint_id]
            
            if joint_type == 3:  # Hinge joint
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                
                # PD control
                pos_error = qpos_ref[qpos_addr] - qpos[qpos_addr]
                vel_error = qvel_ref[qvel_addr] - qvel[qvel_addr]
                
                action[i] = kp * pos_error + kd * vel_error
        
        # Clip to actuator limits
        action = np.clip(action, 
                        self.model.actuator_ctrlrange[:, 0],
                        self.model.actuator_ctrlrange[:, 1])
        
        return action
    
    def compute_reference_trajectory(self, data: mujoco.MjData) -> np.ndarray:
        """
        Compute reference trajectory over MPC horizon.
        Returns array of shape (horizon, nq + nv) containing reference states.
        """
        trajectory = np.zeros((self.config.horizon, self.nq + self.nv))
        
        # For standing, reference is constant standing pose
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        qpos_ref = self.model.key_qpos[keyframe_id]
        qvel_ref = np.zeros(self.nv)
        
        for t in range(self.config.horizon):
            trajectory[t, :self.nq] = qpos_ref
            trajectory[t, self.nq:] = qvel_ref
        
        return trajectory


class G1MPCGuidedEnv(gym.Env):
    """
    MPC-Guided RL Environment for G1 Humanoid.
    
    The RL policy learns to track MPC-generated reference trajectories,
    with rewards based on tracking error and task objectives.
    """
    
    def __init__(self, 
                 render_mode='human', 
                 render_fps=30, 
                 policy_freq=50,
                 external_load_kg: float = 0.0,
                 mpc_config: Optional[MPCConfig] = None):
        super().__init__()
        
        xml_path = "unitree_g1/g1_mocap_29dof_with_hands.xml"
        os.environ.setdefault("MUJOCO_GL", "glfw")
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"\n{'='*50}")
        print(f"MPC-Guided RL Environment")
        print(f"Model: {xml_path}")
        print(f"{'='*50}")
        
        # Dimensions
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # MPC controller
        self.mpc_config = mpc_config or MPCConfig()
        self.mpc = SimpleMPC(self.model, self.mpc_config)
        
        # Action space: RL policy outputs corrections to MPC actions
        action_range = 50.0  # Max correction magnitude
        self.action_space = Box(
            low=-action_range,
            high=action_range,
            shape=(self.nu,),
            dtype=np.float64
        )
        
        # Observation space: [qpos, qvel, mpc_reference_action, tracking_error]
        obs_dim = self.nq + self.nv + self.nu + self.nu
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )
        
        print(f"Action Space (RL corrections): {self.action_space.shape}")
        print(f"Observation Space: {self.observation_space.shape}")
        print(f"  - State (qpos + qvel): {self.nq + self.nv}")
        print(f"  - MPC reference action: {self.nu}")
        print(f"  - Tracking error: {self.nu}")
        print(f"{'='*50}")
        
        # Load standing keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.data.qpos[:] = self.model.key_qpos[keyframe_id]
        self.data.qvel[:] = self.model.key_qvel[keyframe_id]
        mujoco.mj_forward(self.model, self.data)
        
        # Initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
        # Environment parameters
        self.reset_noise = 0.01
        self.min_term_height = 0.4
        self.term_tilt_thresh = np.cos(np.radians(45))
        self.up_vector = np.array([0, 0, 1])
        self.gravity = np.array([0, 0, -9.81])
        self.external_load_mass = max(external_load_kg, 0.0)
        
        # Tracking
        self.mpc_reference_action = np.zeros(self.nu)
        self.previous_action = np.zeros(self.nu)
        
        # Frozen actuators
        self.wrist_and_hand_actuators = list(range(19, 29)) + list(range(33, 43))
        
        # Rendering
        self.viewer = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_dt = 1.0 / self.render_fps
        self.frame_skip = int((1 / self.model.opt.timestep) / policy_freq)
    
    def get_obs(self) -> np.ndarray:
        """Get observation including MPC reference and tracking error"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Compute tracking error (previous MPC action vs actual control)
        tracking_error = self.mpc_reference_action - self.previous_action
        
        obs = np.concatenate([
            qpos,
            qvel,
            self.mpc_reference_action,
            tracking_error
        ])
        
        return obs.astype(np.float64)
    
    def calculate_reward(self) -> float:
        """
        Reward function for MPC-guided RL:
        1. Tracking reward: Follow MPC reference actions
        2. Task reward: Achieve standing objectives
        3. Efficiency: Minimize corrections
        """
        reward = 0.0
        
        # 1. Tracking reward - encourage following MPC reference
        tracking_error = np.linalg.norm(self.mpc_reference_action - self.previous_action)
        tracking_reward = -1.0 * tracking_error
        reward += tracking_reward
        
        # 2. Task objectives (standing)
        height = self.data.qpos[2]
        target_height = 0.79
        height_reward = 5.0 * max(0, 1.0 - abs(height - target_height))
        reward += height_reward
        
        # Upright orientation
        upright_reward = 5.0 * self.get_tilt()
        reward += upright_reward
        
        # 3. Penalize excessive corrections (encourage relying on MPC)
        rl_correction = self.previous_action - self.mpc_reference_action
        correction_penalty = -0.01 * np.sum(np.square(rl_correction))
        reward += correction_penalty
        
        # 4. Stability
        velocity_penalty = -0.01 * np.sum(np.square(self.data.qvel))
        reward += velocity_penalty
        
        # 5. Alive bonus
        reward += 1.0
        
        return reward
    
    def get_tilt(self) -> float:
        """Get upright orientation measure"""
        torso_quat = self.data.qpos[3:7]
        rotation_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_matrix, torso_quat)
        torso_z_axis = rotation_matrix[6:9]
        return np.dot(torso_z_axis, self.up_vector)
    
    def terminate(self) -> bool:
        """Check termination conditions"""
        height = self.data.qpos[2]
        if height < self.min_term_height:
            return True
        if self.get_tilt() < self.term_tilt_thresh:
            return True
        return False
    
    def apply_external_load(self):
        """Apply external load if configured"""
        if self.external_load_mass <= 0:
            return
        
        load_force = self.gravity * self.external_load_mass
        force_per_hand = load_force / 2
        
        hand_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        for hand_name in hand_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
            hand_pos = self.data.xpos[body_id].copy()
            
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_per_hand,
                np.zeros(3),
                hand_pos,
                body_id,
                self.data.qfrc_applied
            )
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset to standing with noise
        noise = self.reset_noise * np.random.randn(self.nq)
        noise[3:7] = 0  # Don't perturb quaternion
        self.data.qpos[:] = self.init_qpos + noise
        self.data.qvel[:] = self.init_qvel + self.reset_noise * np.random.randn(self.nv)
        
        mujoco.mj_forward(self.model, self.data)
        
        # Compute initial MPC reference
        self.mpc_reference_action = self.mpc.compute_reference_action(self.data)
        self.previous_action = self.mpc_reference_action.copy()
        
        obs = self.get_obs()
        info = {'mpc_active': True}
        
        return obs, info
    
    def step(self, rl_correction: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step with MPC + RL correction.
        
        Args:
            rl_correction: RL policy output (corrections to MPC action)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Clear applied forces
        self.data.qfrc_applied[:] = 0
        
        # 1. Get MPC reference action
        self.mpc_reference_action = self.mpc.compute_reference_action(self.data)
        
        # 2. Apply RL correction
        final_action = self.mpc_reference_action + rl_correction
        
        # 3. Clip to actuator limits
        final_action = np.clip(final_action,
                              self.model.actuator_ctrlrange[:, 0],
                              self.model.actuator_ctrlrange[:, 1])
        
        # 4. Freeze hands
        final_action[self.wrist_and_hand_actuators] = 0
        
        # 5. Apply action
        self.data.ctrl[:] = final_action
        self.previous_action = final_action.copy()
        
        # 6. Apply external load if configured
        if self.external_load_mass > 0:
            self.apply_external_load()
        
        # 7. Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # 8. Get results
        obs = self.get_obs()
        reward = self.calculate_reward()
        terminated = self.terminate()
        truncated = False
        
        info = {
            'mpc_reference': self.mpc_reference_action.copy(),
            'rl_correction': rl_correction.copy(),
            'final_action': final_action.copy(),
            'tracking_error': np.linalg.norm(final_action - self.mpc_reference_action),
            'height': self.data.qpos[2],
            'upright': self.get_tilt()
        }
        
        # 9. Render
        if self.render_mode == 'human':
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
    
    def close(self):
        """Clean up"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
