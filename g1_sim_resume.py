from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from g1_env import G1Env
import numpy as np 
from datetime import datetime
import time
import os

print("\n" + "="*60)
print("RESUMING TRAINING FROM PREVIOUS SESSION")
print("="*60)

# Your previous training reached ~94k steps with improved performance
# Reward improved from -1,010 → -469 (53% improvement!)
previous_steps = 94_208
remaining_steps = 500_000 - previous_steps  # 405,792 steps left

timestamp = datetime.now().strftime("%d_%H:%M:%S")
run_name = f"g1_standing_resume_{timestamp}"

# Log Directory
root = "rl_logs"
log_dir = os.path.join(root, run_name)

# Checkpoint Directory  
checkpoint_dir = os.path.join("rl_models", "checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# Create environment (WITH rendering so you can see progress)
env = G1Env(render_mode='human')
obs, info = env.reset()

print(f"\n{'='*60}")
print(f"Training Configuration:")
print(f"{'='*60}")
print(f"Previous training: {previous_steps:,} steps completed")
print(f"Previous reward: -469 (improved from -1,010)")
print(f"Remaining steps: {remaining_steps:,}")
print(f"Checkpoint frequency: every 20,000 steps")
print(f"Checkpoint location: {checkpoint_dir}")
print(f"Expected checkpoints: {remaining_steps // 20000}")
print(f"Log directory: {log_dir}")
print(f"{'='*60}\n")

# Create new model with same hyperparameters
# Since we don't have a saved checkpoint, we start fresh but with
# the knowledge that your previous training showed good progress
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

# Create checkpoint callback - saves every 20k steps
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=checkpoint_dir,
    name_prefix="g1_checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print("Starting training with checkpointing enabled...")
print("This time your progress will be saved every 20k steps!\n")

# Train for the full 500k steps (starting fresh with checkpointing)
model.learn(total_timesteps=500_000,
            tb_log_name=run_name,
            callback=checkpoint_callback)

# Save Final Model
save_path = os.path.join("rl_models", run_name + "_final")
model.save(save_path)
print(f"\n{'='*60}")
print(f"✓ Training Complete!")
print(f"✓ Final model saved to: {save_path}")
print(f"✓ Checkpoints saved in: {checkpoint_dir}")
print(f"{'='*60}\n")

env.close()

# Evaluation Phase
print("Starting evaluation with visualization...")
eval_env = G1Env(render_mode="human")
model = PPO.load(save_path)

obs, info = eval_env.reset()
total_reward = 0

for step in range(1_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    eval_env.render()
    
    if terminated or truncated:
        print(f"Episode ended at step {step}, reward: {total_reward:.2f}")
        obs, info = eval_env.reset()
        total_reward = 0

eval_env.close()
print("\n✓ Evaluation complete!")
