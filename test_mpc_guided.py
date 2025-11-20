"""
Test script for MPC-guided RL policy.

Visualizes the trained policy tracking MPC reference actions.
"""

import numpy as np
import argparse
from stable_baselines3 import PPO
from g1_mpc_guided_env import G1MPCGuidedEnv, MPCConfig
import time


def test_policy(model_path: str, n_episodes: int = 5, render: bool = True):
    """
    Test a trained MPC-guided RL policy.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to test
        render: Whether to render visualization
    """
    
    print("\n" + "="*70)
    print("TESTING MPC-GUIDED RL POLICY")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print("="*70 + "\n")
    
    # Load trained model
    print("Loading model...")
    model = PPO.load(model_path)
    
    # Create environment
    print("Creating environment...")
    env = G1MPCGuidedEnv(
        render_mode='human' if render else None,
        render_fps=30,
        policy_freq=50,
        external_load_kg=0.0,
        mpc_config=MPCConfig(
            horizon=20,
            dt=0.02,
            target_height=0.79
        )
    )
    
    # Test episodes
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_tracking_errors = []
        
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get RL correction
            rl_correction, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(rl_correction)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            episode_tracking_errors.append(info['tracking_error'])
            
            # Print info every 100 steps
            if episode_length % 100 == 0:
                print(f"  Step {episode_length}:")
                print(f"    Height: {info['height']:.3f}m")
                print(f"    Upright: {info['upright']:.3f}")
                print(f"    Tracking Error: {info['tracking_error']:.3f}")
                print(f"    RL Correction Magnitude: {np.linalg.norm(info['rl_correction']):.3f}")
            
            if render:
                time.sleep(0.02)  # Match control frequency
        
        # Episode summary
        avg_tracking_error = np.mean(episode_tracking_errors)
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Avg Tracking Error: {avg_tracking_error:.3f}")
        print(f"  Termination: {'terminated' if terminated else 'truncated'}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        tracking_errors.append(avg_tracking_error)
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Average Tracking Error: {np.mean(tracking_errors):.3f} ± {np.std(tracking_errors):.3f}")
    print("="*70 + "\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test MPC-guided RL policy")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (e.g., rl_models/g1_mpc_guided_XX_XX:XX:XX/best_model.zip)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of test episodes (default: 5)'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable visualization'
    )
    
    args = parser.parse_args()
    
    test_policy(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
