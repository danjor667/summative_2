#!/usr/bin/env python3

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper

from utils.helpers import TrainingCallback


def create_environment(env_id="BreakoutNoFrameskip-v4", render_mode=None):
    """Create and wrap an Atari environment with error handling."""
    import ale_py

    try:
        # Try to create the environment
        env = gym.make(env_id, render_mode=render_mode)

        # Apply AtariWrapper
        from stable_baselines3.common.atari_wrappers import AtariWrapper

        env = AtariWrapper(env)

        return env
    except Exception as e:
        print(f"Error creating environment {env_id}: {e}")
        print("\nTrying to list available Atari environments:")
        try:
            # List available environments
            all_envs = gym.envs.registry.keys()
            atari_envs = [
                env for env in all_envs if "ALE" in env or "NoFrameskip" in env
            ]
            if atari_envs:
                print(
                    f"Available Atari environments: {atari_envs[:5]} ... (and {len(atari_envs)-5} more)"
                )
                print(f"\nTry using one of these instead of {env_id}")
            else:
                print(
                    "No Atari environments found. Make sure you have installed gymnasium[atari,accept-rom-license]"
                )
        except Exception as list_error:
            print(f"Error listing environments: {list_error}")

        # Return a simple CartPole environment as fallback
        print("\nFalling back to CartPole-v1 environment for demonstration purposes")
        return gym.make("CartPole-v1", render_mode=render_mode)


def train_agent(
    env,
    policy_type="CnnPolicy",
    hyperparams=None,
    timesteps=2000000,
    experiment_name="default",
):
    """Train a DQN agent with the specified policy and hyperparameters."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with Monitor
    env = Monitor(env, log_dir)

    # Default hyperparameters
    default_params = {
        "learning_rate": 1e-4,
        "buffer_size": 500000,
        "learning_starts": 10000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "train_freq": 8,
        "gradient_steps": 2,
        "target_update_interval": 2000,
    }

    # Update with provided hyperparameters
    if hyperparams:
        default_params.update(hyperparams)

    print(f"Training DQN agent with {policy_type} policy")
    print(f"Hyperparameters: {default_params}")

    # Create the DQN agent
    model = DQN(
        policy_type,
        env,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{experiment_name}/",
        **default_params,
    )

    # Create callback
    callback = TrainingCallback(plot_interval=10)

    # Train the agent
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save the trained model
    model_path = f"dqn_model_{experiment_name}"
    model.save(model_path)
    print(f"Model saved as {model_path}.zip")

    # Print final training statistics
    final_stats = {}
    if callback.rewards:
        avg_reward = (
            np.mean(callback.rewards[-100:])
            if len(callback.rewards) >= 100
            else np.mean(callback.rewards)
        )
        avg_length = (
            np.mean(callback.episode_lengths[-100:])
            if len(callback.episode_lengths) >= 100
            else np.mean(callback.episode_lengths)
        )
        print(f"Final training statistics:")
        print(f"Total episodes: {len(callback.rewards)}")
        print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
        print(f"Average episode length over last 100 episodes: {avg_length:.2f}")

        final_stats = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "total_episodes": len(callback.rewards),
            "rewards": callback.rewards,
            "episode_lengths": callback.episode_lengths,
        }

    return model, final_stats


def main():
    # Create a fresh environment
    env_id = "BreakoutNoFrameskip-v4"
    env = create_environment(env_id)

    # Check if we're using CartPole (fallback) or an Atari environment
    is_atari = "ALE" in str(env) or "NoFrameskip" in env_id
    policy_type = "CnnPolicy" if is_atari else "MlpPolicy"

    print(f"Using {policy_type} for environment {env_id}")

    # Train the agent (adjust timesteps as needed)
    model, training_stats = train_agent(
        env,
        policy_type=policy_type,
        experiment_name="default_run",
    )


if __name__ == "__main__":
    main()
