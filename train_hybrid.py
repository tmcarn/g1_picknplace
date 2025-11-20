"""
Train G1 with Hybrid PD + RL Controller
Much faster learning because RL only learns small corrections!
"""
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_hybrid_env import G1HybridEnv
from datetime import datetime
import os

print("\n" + "="*60)
print("HYBRID PD + RL TRAINING")
print("="*60)
print("Strategy: PD controller provides baseline stability")
print("          RL learns small corrections on top")
print("Advantage: MUCH faster learning (smaller action space)")
print("="*60 + "\n")

# Configuration
timestamp = datetime.now().strftime("%d_%H:%M:%S")
n_steps = 100_000  # Should learn much faster!
run_name = f"g1_hybrid_{n_steps}_steps_{timestamp}"

# Directories
log_dir = os.path.join("rl_logs", run_name)
checkpoint_dir = os.path.join("rl_models", "checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# Create hybrid environment (with rendering)
env = G1HybridEnv(render_mode='human')
obs, info = env.reset()

# PPO with optimized hyperparameters
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
    save_freq=10000,
    save_path=checkpoint_dir,
    name_prefix="g1_hybrid_checkpoint",
)

print(f"Training Configuration:")
print(f"  Total timesteps: {n_steps:,}")
print(f"  Checkpoint frequency: every 10,000 steps")
print(f"  Checkpoint location: {checkpoint_dir}")
print(f"  Log directory: {log_dir}")
print(f"\nStarting training...")
print("Expected to learn MUCH faster than pure RL!\n")

# Train
model.learn(total_timesteps=n_steps,
            tb_log_name=run_name,
            callback=checkpoint_callback)

# Save final model
save_path = os.path.join("rl_models", run_name + "_final")
model.save(save_path)
print(f"\n✓ Training complete!")
print(f"✓ Final model saved to: {save_path}\n")

env.close()

# Evaluation with visualization
print("="*60)
print("EVALUATION PHASE")
print("="*60)
print("Loading trained model and running with visualization...\n")

eval_env = G1HybridEnv(render_mode="human")
model = PPO.load(save_path)

obs, info = eval_env.reset()
total_reward = 0
episode_count = 0

for step in range(1_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    eval_env.render()
    
    if step % 100 == 0:
        print(f"Step {step:4d} | Height: {info['height']:.3f}m | "
              f"Reward: {reward:7.2f} | "
              f"PD norm: {info['pd_control_norm']:.1f} | "
              f"RL norm: {info['rl_correction_norm']:.1f}")
    
    if terminated or truncated:
        episode_count += 1
        print(f"\nEpisode {episode_count} ended | Total reward: {total_reward:.2f}")
        obs, info = eval_env.reset()
        total_reward = 0

eval_env.close()
print("\n✓ Evaluation complete!")
print("="*60)
