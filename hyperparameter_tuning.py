#!/usr/bin/env python3
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper

from train import create_environment


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []

    def _on_step(self):
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.rewards) % 10 == 0:
                    avg_reward = np.mean(self.rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"Episode: {len(self.rewards)}")
                    print(f"Average reward over last 10 episodes: {avg_reward:.2f}")
                    print(
                        f"Average episode length over last 10 episodes: {avg_length:.2f}"
                    )
                    print("-" * 50)
        return True


def train_with_hyperparams(hyperparams, experiment_name, timesteps=100000):
    """Train a DQN agent with the given hyperparameters."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env_id = "BreakoutNoFrameskip-v4"
    env = create_environment(env_id)
    env = gym.make(env_id, render_mode=None)
    env = AtariWrapper(env)
    env = Monitor(env, log_dir)

    print(f"Training DQN agent for experiment: {experiment_name}")
    print(f"Hyperparameters: {hyperparams}")

    # Create the DQN agent
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{experiment_name}/",
        **hyperparams,
    )

    # Create callback
    callback = TrainingCallback()

    # Train the agent
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save the trained model
    model_path = f"dqn_model_{experiment_name}"
    model.save(model_path)
    print(f"Model saved as {model_path}.zip")

    # Print final training statistics
    final_stats = {}
    if callback.rewards:
        avg_reward = np.mean(callback.rewards[-100:])
        avg_length = np.mean(callback.episode_lengths[-100:])
        print(f"Final training statistics:")
        print(f"Total episodes: {len(callback.rewards)}")
        print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
        print(f"Average episode length over last 100 episodes: {avg_length:.2f}")

        final_stats = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "total_episodes": len(callback.rewards),
        }

    env.close()
    return final_stats


def main():
    # Define different hyperparameter sets to test
    hyperparameter_sets = [
        {
            "name": "high_lr",
            "params": {
                "learning_rate": 5e-4,  # Higher learning rate
                "buffer_size": 500000,
                "learning_starts": 10000,
                "batch_size": 32,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
            },
        },
        {
            "name": "high_gamma",
            "params": {
                "learning_rate": 1e-4,
                "buffer_size": 500000,
                "learning_starts": 10000,
                "batch_size": 32,
                "gamma": 0.995,  # Higher discount factor
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
            },
        },
        {
            "name": "large_batch",
            "params": {
                "learning_rate": 1e-4,
                "buffer_size": 500000,
                "learning_starts": 10000,
                "batch_size": 64,  # Larger batch size
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
            },
        },
    ]

    # Train with each hyperparameter set and collect results
    results = []
    for hp_set in hyperparameter_sets:
        print(f"\n{'='*50}")
        print(f"Starting experiment: {hp_set['name']}")
        print(f"{'='*50}\n")

        stats = train_with_hyperparams(
            hp_set["params"],
            hp_set["name"],
            timesteps=2000000,
        )

        results.append(
            {"name": hp_set["name"], "hyperparams": hp_set["params"], "stats": stats}
        )

    # Print summary of all experiments
    print("\n\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    print(
        f"{'Experiment':<15} | {'Learning Rate':<15} | {'Gamma':<8} | {'Batch Size':<10} | {'Avg Reward':<15}"
    )
    print("-" * 80)

    for result in results:
        hp = result["hyperparams"]
        stats = result["stats"]
        print(
            f"{result['name']:<15} | {hp['learning_rate']:<15.6f} | {hp['gamma']:<8.3f} | {hp['batch_size']:<10} | {stats.get('avg_reward', 'N/A'):<15}"
        )


if __name__ == "__main__":
    main()
