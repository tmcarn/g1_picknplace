"""
Run MPC-guided environment with rendering (no training).
This lets you see the MPC controller in action.
"""

import numpy as np
from g1_mpc_guided_env import G1MPCGuidedEnv, MPCConfig
import time


def run_mpc_guided_env():
    """Run environment with MPC controller and random RL corrections"""
    
    print("\n" + "="*70)
    print("MPC-GUIDED ENVIRONMENT - INTERACTIVE DEMO")
    print("="*70)
    print("This runs the MPC controller with random RL corrections")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Create environment with rendering
    env = G1MPCGuidedEnv(
        render_mode='human',
        render_fps=30,
        policy_freq=50,
        external_load_kg=0.0,
        mpc_config=MPCConfig(
            horizon=20,
            dt=0.02,
            target_height=0.79
        )
    )
    
    try:
        # Reset environment
        obs, info = env.reset()
        print("Environment reset. Starting simulation...\n")
        
        step = 0
        episode_reward = 0
        
        while True:
            # Random RL correction (small noise around zero)
            # In trained policy, this would be the learned correction
            rl_correction = np.random.randn(env.nu) * 1.0
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(rl_correction)
            
            episode_reward += reward
            step += 1
            
            # Print status every 100 steps
            if step % 100 == 0:
                print(f"Step {step}:")
                print(f"  Height: {info['height']:.3f}m (target: 0.79m)")
                print(f"  Upright: {info['upright']:.3f}")
                print(f"  Tracking Error: {info['tracking_error']:.3f}")
                print(f"  Episode Reward: {episode_reward:.2f}")
                print()
            
            # Reset if terminated
            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
                print("\nResetting...\n")
                
                obs, info = env.reset()
                step = 0
                episode_reward = 0
            
            # Small delay to match control frequency
            time.sleep(0.02)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    run_mpc_guided_env()
