from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from g1_env import G1Env
import numpy as np 
from datetime import datetime
import time
import os

# Define Run Name
timestamp = datetime.now().strftime("%d_%H:%M:%S")
n_steps = 100_000
run_name = f"g1_standing_{n_steps}_steps_{timestamp}"

# Log Directory
root = "rl_logs"
log_dir = os.path.join(root, run_name)

# Train Model
env = G1Env(render_mode=None)
obs, info = env.reset()
model = PPO(MlpPolicy, 
            env, 
            verbose=1,
            tensorboard_log=log_dir)

model.learn(total_timesteps=n_steps,
            tb_log_name=run_name)

# Save Model
save_path = os.path.join("rl_models", run_name)
model.save(save_path)

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


