from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_env import G1Env
import numpy as np 
from datetime import datetime
import time
import os

# Define Run Name
timestamp = datetime.now().strftime("%d_%H:%M:%S")
n_steps = 500_000  # Increased training steps for better learning
run_name = f"g1_standing_{n_steps}_steps_{timestamp}"

# Log Directory
root = "rl_logs"
log_dir = os.path.join(root, run_name)

# Checkpoint Directory
checkpoint_dir = os.path.join("rl_models", "checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# Train Model
env = G1Env(render_mode=None)  # No rendering during training for speed
obs, info = env.reset()

# Improved PPO hyperparameters for better learning
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
            ent_coef=0.01,  # Encourage exploration
            tensorboard_log=log_dir)

# Create checkpoint callback - saves every 20k steps
checkpoint_callback = CheckpointCallback(
    save_freq=20000,  # Save every 20,000 steps
    save_path=checkpoint_dir,
    name_prefix="g1_checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print(f"\n{'='*60}")
print(f"Training Configuration:")
print(f"{'='*60}")
print(f"Total timesteps: {n_steps:,}")
print(f"Checkpoint frequency: every 20,000 steps")
print(f"Checkpoint location: {checkpoint_dir}")
print(f"Expected checkpoints: {n_steps // 20000}")
print(f"Log directory: {log_dir}")
print(f"{'='*60}\n")

model.learn(total_timesteps=n_steps,
            tb_log_name=run_name,
            callback=checkpoint_callback)  # Add callback here

# Save Final Model
save_path = os.path.join("rl_models", run_name)
model.save(save_path)
print(f"\n✓ Final model saved to: {save_path}")

env.close()

# Load model
eval_env = G1Env(render_mode="human")
model = PPO.load(save_path)

# Evaluate
obs, info = eval_env.reset()
for step in range(1_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    
    # if terminated or truncated:
    #     obs, info = eval_env.reset()

eval_env.close()


