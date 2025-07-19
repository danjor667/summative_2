#!/usr/bin/env python3
import time
import numpy as np
from typing import Optional

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.atari_wrappers import AtariWrapper
except ImportError:
    DQN = None
    AtariWrapper = None
try:
    import gymnasium as gym
except ImportError:
    gym = None
try:
    import ale_py
except ImportError:
    ale_py = None


def create_environment(env_id="BreakoutNoFrameskip-v4", render_mode=None):
    """Create and wrap an Atari environment with error handling."""
    import ale_py
    import gymnasium as gym

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
            from gymnasium.envs.registration import registry

            all_envs = registry.keys()
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


def main():
    pass
    # TODO


def evaluate_agent(model, env_id, num_episodes=5, render_mode: Optional[str] = "human"):
    """Evaluate a trained agent."""
    try:
        # Create environment with rendering
        env = create_environment(env_id, render_mode=render_mode)

        rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False

            print(f"Starting episode {episode+1}/{num_episodes}")

            while not (done or truncated):
                # Get action from the trained model
                action, _ = model.predict(obs, deterministic=True)

                # Execute action in the environment
                obs, reward, done, truncated, info = env.step(action)

                # Update episode statistics
                episode_reward += float(reward)
                episode_length += 1

                # Slow down rendering to make it viewable
                if render_mode == "human":
                    time.sleep(0.01)

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(
                f"Episode {episode+1} finished with reward: {episode_reward}, length: {episode_length}"
            )

        # Print evaluation statistics
        print("\nEvaluation Results:")
        print(f"Average reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print(f"Rewards per episode: {rewards}")

        env.close()
        return rewards, episode_lengths

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Falling back to non-rendering evaluation")

        # Try again without rendering
        if render_mode == "human":
            return evaluate_agent(model, env_id, num_episodes, render_mode="rgb_array")
        return [], []


if __name__ == "__main__":
    main()
