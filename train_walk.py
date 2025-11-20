import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
from g1_walk_env import G1WalkEnv, G1WalkConfig

# Create directories for logs and models
log_dir = "rl_logs/"
model_dir = "rl_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Configure walking task
config = G1WalkConfig(
    target_velocity=0.5,  # Start with moderate walking speed
    forward_velocity_weight=10.0,
    forward_progress_weight=30.0,
    backward_penalty_weight=15.0,
    stand_penalty_weight=10.0,
    height_weight=3.0,
    upright_weight=5.0,
    stability_weight=2.0,
    gait_symmetry_weight=1.0,
    foot_clearance_weight=2.0,
    gait_phase_weight=1.0,
    smooth_motion_weight=0.5,
    energy_penalty_weight=0.001,
    torque_penalty_weight=0.0001,
    initial_forward_velocity=0.3,
)

# Create the environment with rendering enabled
def make_env():
    return G1WalkEnv(render_mode='human', config=config)

env = make_vec_env(make_env, n_envs=1)

# --- PPO Model ---
# Define the PPO model with tuned hyperparameters for walking
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Slightly higher entropy for exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

# --- Callbacks ---
# Callback for saving the best model
eval_callback = EvalCallback(
    env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=500,
    deterministic=True,
    render=False  # Disable rendering during evaluation
)

# --- Training ---
# Train the model
print("\n" + "="*50)
print("Starting training for walking task")
print("Target: Learn to walk forward at 0.5 m/s")
print("Key objectives:")
print("  - Maintain forward velocity")
print("  - Keep upright posture")
print("  - Develop natural gait pattern")
print("  - Stay stable and efficient")
print("="*50 + "\n")

model.learn(total_timesteps=300000, callback=eval_callback)

# --- Save the final model ---
model.save(f"{model_dir}/g1_walk_final")

# --- Close the environment ---
env.close()

print("\nTraining completed!")
print(f"Best model saved to: {model_dir}/best_model.zip")
print(f"Final model saved to: {model_dir}/g1_walk_final.zip")
