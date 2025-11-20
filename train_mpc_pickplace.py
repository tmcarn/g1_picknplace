"""
Train G1 with MPC + RL for Pick and Place
MPC handles trajectory planning, RL learns corrections
"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_mpc_pickplace_env import G1MPCPickPlaceEnv
import os
from datetime import datetime

print("\n" + "="*60)
print("MPC + RL PICK AND PLACE TRAINING")
print("="*60)
print("MPC provides:")
print("  ✓ Trajectory planning (reach → grasp → lift → move → place)")
print("  ✓ Inverse kinematics for arm control")
print("  ✓ Standing stability control")
print("")
print("RL learns:")
print("  ✓ Small corrections to improve MPC performance")
print("  ✓ Adaptation to different box/target positions")
print("  ✓ Fine-tuning for optimal execution")
print("="*60 + "\n")

# Configuration
timestamp = datetime.now().strftime("%d_%H:%M:%S")
n_steps = 100_000  # Much less than pure RL (500k)!
run_name = f"g1_mpc_pickplace_{n_steps}_steps_{timestamp}"

# Directories
log_dir = os.path.join("rl_logs", run_name)
checkpoint_dir = os.path.join("rl_models", "checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# Create environment with rendering
env = G1MPCPickPlaceEnv(render_mode='human')
obs, info = env.reset()

print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape} (corrections only!)")
print(f"Number of actuators: {env.model.nu}")

# PPO with smaller network (action space is just corrections)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,  # Lower LR for fine-tuning
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log=log_dir
)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=checkpoint_dir,
    name_prefix="mpc_pickplace"
)

print(f"\nTraining Configuration:")
print(f"  Total timesteps: {n_steps:,}")
print(f"  Checkpoint frequency: every 10,000 steps")
print(f"  Checkpoint location: {checkpoint_dir}")
print(f"  Log directory: {log_dir}")
print(f"\nStarting training...")
print("Expected training time: ~1-2 hours (5x faster than pure RL!)")
print("Watch MPC plan the motions and RL refine them!\n")

# Train
model.learn(
    total_timesteps=n_steps,
    callback=checkpoint_callback,
    progress_bar=True
)

# Save final model
final_model_path = os.path.join("rl_models", f"{run_name}_final.zip")
model.save(final_model_path)

print(f"\n{'='*60}")
print("Training Complete!")
print(f"Final model saved to: {final_model_path}")
print(f"{'='*60}\n")

env.close()
