#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress."""

    def __init__(self, verbose=0, plot_interval=10):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []
        self.plot_interval = plot_interval

    def _on_step(self):
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.rewards) % self.plot_interval == 0:
                    avg_reward = np.mean(self.rewards[-self.plot_interval :])
                    avg_length = np.mean(self.episode_lengths[-self.plot_interval :])
                    print(f"Episode: {len(self.rewards)}")
                    print(
                        f"Average reward over last {self.plot_interval} episodes: {avg_reward:.2f}"
                    )
                    print(
                        f"Average episode length over last {self.plot_interval} episodes: {avg_length:.2f}"
                    )
                    print("-" * 50)

                    # Plot training progress
                    if (
                        len(self.rewards) >= 20
                    ):  # Only plot after collecting enough data
                        self.plot_training_progress()
        return True

    def plot_training_progress(self):
        """Plot the training progress."""
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards
        ax1.plot(self.rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot moving average of rewards
        window_size = min(20, len(self.rewards))
        moving_avg = np.convolve(
            self.rewards, np.ones(window_size) / window_size, mode="valid"
        )
        ax1.plot(
            range(window_size - 1, len(self.rewards)),
            moving_avg,
            "r-",
            label=f"{window_size}-episode moving average",
        )
        ax1.legend()

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Length")

        plt.tight_layout()
        plt.show()
