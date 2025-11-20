"""
Training script for MPC-guided RL policy on G1 humanoid.

The RL policy learns to:
1. Track MPC reference actions
2. Correct for model mismatch
3. Improve upon MPC performance
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from datetime import datetime

# Import the MPC-guided environment
from g1_mpc_guided_env import G1MPCGuidedEnv, MPCConfig


def make_env(rank=0):
    """Create environment factory"""
    env = G1MPCGuidedEnv(
        render_mode=True,  # No rendering during training
        render_fps=30,
        policy_freq=50,
        external_load_kg=0.0,
        mpc_config=MPCConfig(
            horizon=20,
            dt=0.02,
            state_weight=1.0,
            control_weight=0.01,
            reference_weight=10.0,
            target_height=0.79
        )
    )
    return env


def train_mpc_guided_policy():
    """Train MPC-guided RL policy"""
    
    print("\n" + "="*70)
    print("MPC-GUIDED RL TRAINING FOR G1 HUMANOID")
    print("="*70)
    print("\nTraining Philosophy:")
    print("- MPC provides reference actions based on model")
    print("- RL learns to correct MPC actions for:")
    print("  1. Model mismatch (sim-to-real)")
    print("  2. Unmodeled dynamics")
    print("  3. Task-specific optimization")
    print("="*70 + "\n")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%y_%H:%M:%S")
    run_name = f"g1_mpc_guided_{timestamp}"
    
    # Directories
    log_dir = f"rl_logs/{run_name}"
    model_dir = f"rl_models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}\n")
    
    # Create parallel environments for faster training
    n_envs = 4
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = G1MPCGuidedEnv(
        render_mode=None,
        render_fps=30,
        policy_freq=50,
        external_load_kg=0.0
    )
    
    # PPO hyperparameters
    # Using slightly different hyperparameters for tracking task
    print("\nPPO Hyperparameters:")
    hyperparams = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,  # No entropy bonus for tracking
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        'verbose': 1,
        'tensorboard_log': log_dir
    }
    
    for key, value in hyperparams.items():
        if key not in ['policy', 'env', 'policy_kwargs']:
            print(f"  {key}: {value}")
    print(f"  policy_kwargs: {hyperparams['policy_kwargs']}")
    
    print("\nInitializing PPO agent...")
    model = PPO(**hyperparams)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,  # Evaluate every 5000 steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="checkpoint"
    )
    
    # Training
    total_timesteps = 200000  # 200k steps for tracking task
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"With {n_envs} parallel environments, this is {total_timesteps // n_envs:,} iterations\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Best model saved to: {model_dir}/best_model.zip")
    print(f"Tensorboard logs: {log_dir}")
    print(f"{'='*70}\n")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\nTo test the trained policy:")
    print(f"  python test_mpc_guided.py --model {model_dir}/best_model.zip")
    print()


if __name__ == "__main__":
    train_mpc_guided_policy()
