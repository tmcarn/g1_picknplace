import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
from g1_squat_env import G1SquatEnv

# Create directories for logs and models
log_dir = "rl_logs/"
model_dir = "rl_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create the environment with rendering enabled
def make_env():
    return G1SquatEnv(render_mode='human')

env = make_vec_env(make_env, n_envs=1)

# --- PPO Model ---
# Define the PPO model with tuned hyperparameters
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
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
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
print("Starting training for squatting task")
print("Target: Learn to bend down and stand up smoothly")
print("="*50 + "\n")

model.learn(total_timesteps=100000, callback=eval_callback)

# --- Save the final model ---
model.save(f"{model_dir}/g1_squat_final")

# --- Close the environment ---
env.close()

print("\nTraining completed!")
print(f"Best model saved to: {model_dir}/best_model.zip")
print(f"Final model saved to: {model_dir}/g1_squat_final.zip")
