#!/usr/bin/env python3
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper


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


def train_with_policy(policy_type, experiment_name, timesteps=100000):
    """Train a DQN agent with the specified policy type."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env_id = "BreakoutNoFrameskip-v4"
    env = gym.make(env_id, render_mode=None)
    env = AtariWrapper(env)
    env = Monitor(env, log_dir)

    print(f"Training DQN agent with {policy_type} policy")

    # Default hyperparameters
    hyperparams = {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 10000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
    }

    # Create the DQN agent with specified policy
    model = DQN(
        policy_type,
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
    # Define policies to compare
    policies = [
        {"type": "MlpPolicy", "name": "mlp_policy"},
        {"type": "CnnPolicy", "name": "cnn_policy"},
    ]

    # Train with each policy and collect results
    results = []
    for policy in policies:
        print(f"\n{'='*50}")
        print(f"Starting experiment with {policy['type']}")
        print(f"{'='*50}\n")

        stats = train_with_policy(
            policy["type"],
            policy["name"],
            timesteps=100000,  # Reduced for quicker experiments
        )

        results.append(
            {"policy": policy["type"], "name": policy["name"], "stats": stats}
        )

    # Print summary of policy comparison
    print("\n\n" + "=" * 60)
    print("POLICY COMPARISON RESULTS")
    print("=" * 60)
    print(
        f"{'Policy':<15} | {'Avg Reward':<15} | {'Avg Episode Length':<20} | {'Total Episodes':<15}"
    )
    print("-" * 60)

    for result in results:
        stats = result["stats"]
        print(
            f"{result['policy']:<15} | {stats.get('avg_reward', 'N/A'):<15.2f} | {stats.get('avg_length', 'N/A'):<20.2f} | {stats.get('total_episodes', 'N/A'):<15}"
        )


if __name__ == "__main__":
    main()
