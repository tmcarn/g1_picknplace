"""
Train G1 for Pick and Place Task
"""
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_pickplace_env import G1PickPlaceEnv
from datetime import datetime
import os

print("\n" + "="*60)
print("G1 PICK AND PLACE TRAINING")
print("="*60)
print("Task: Pick up box and place it at target location")
print("Reward components:")
print("  - Stay upright")
print("  - Reach for box")
print("  - Lift box")
print("  - Move box to target")
print("  - Place at target")
print("="*60 + "\n")

# Configuration
timestamp = datetime.now().strftime("%d_%H:%M:%S")
n_steps = 500_000  # This is a complex task, needs more training
run_name = f"g1_pickplace_{n_steps}_steps_{timestamp}"

# Directories
log_dir = os.path.join("rl_logs", run_name)
checkpoint_dir = os.path.join("rl_models", "checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# Create environment with rendering
env = G1PickPlaceEnv(render_mode='human')
obs, info = env.reset()

print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")
print(f"Number of actuators: {env.model.nu}")

# PPO with optimized hyperparameters for complex manipulation
model = PPO(MlpPolicy, 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=log_dir)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=checkpoint_dir,
    name_prefix="g1_pickplace"
)

print(f"\nTraining Configuration:")
print(f"  Total timesteps: {n_steps:,}")
print(f"  Checkpoint frequency: every 20,000 steps")
print(f"  Checkpoint location: {checkpoint_dir}")
print(f"  Log directory: {log_dir}")
print(f"\nStarting training...")
print("This is a complex task - expect longer training time!")
print("Watch the robot learn to reach, grasp, lift, and place!\n")

# Train
model.learn(total_timesteps=n_steps,
            callback=checkpoint_callback,
            progress_bar=True)

# Save final model
final_model_path = os.path.join("rl_models", f"{run_name}_final.zip")
model.save(final_model_path)

print(f"\n{'='*60}")
print("Training Complete!")
print(f"Final model saved to: {final_model_path}")
print(f"{'='*60}\n")

env.close()
