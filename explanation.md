# DQN Agent for Atari Games - Implementation Explanation

This document explains the implementation details of the DQN agent for playing Atari games, focusing on the training process, evaluation, hyperparameter tuning, and policy comparison.

## Training Process (`train.py`)

### Why DQN for Atari Games?

Deep Q-Networks (DQN) are particularly well-suited for Atari games because:

1. **Image-based Input**: DQN can process raw pixel data through convolutional layers
2. **Experience Replay**: Stores and reuses past experiences to break correlations between consecutive samples
3. **Fixed Q-Targets**: Uses a separate target network to reduce overestimation bias and improve stability

### Implementation Details

- **Environment**: We use `BreakoutNoFrameskip-v4` wrapped with `AtariWrapper` which:
  - Performs frame skipping (repeating actions)
  - Applies frame stacking (combining multiple frames)
  - Normalizes observations
  - Clips rewards to {-1, 0, 1}

- **CNN Policy**: For image-based Atari games, CNN policy is more effective as it can extract spatial features from the game frames.

- **Callback System**: We implemented a custom callback to track and report training progress, including:
  - Episode rewards
  - Episode lengths
  - Running averages for monitoring improvement

## Playing with the Agent (`play.py`)

The play script demonstrates the trained agent's performance by:

1. Loading the saved model
2. Running it in the environment with rendering enabled
3. Collecting and reporting performance metrics

This allows us to visually verify that the agent has learned effective strategies for the game.

## Hyperparameter Tuning (`hyperparameter_tuning.py`)

### Why Tune Hyperparameters?

Hyperparameter tuning is crucial for DQN performance because:

1. **Learning Rate**: Controls how quickly the network adapts to new information
   - Too high: Can cause unstable training and divergence
   - Too low: Can result in slow learning and getting stuck in local optima

2. **Gamma (Discount Factor)**: Determines the importance of future rewards
   - Higher values (closer to 1): Agent values future rewards more
   - Lower values: Agent becomes more myopic, focusing on immediate rewards

3. **Batch Size**: Affects training stability and speed
   - Larger batches: More stable gradient estimates but slower training
   - Smaller batches: Faster training but noisier updates

4. **Exploration Parameters**: Balance exploration vs. exploitation
   - Initial epsilon: Starting exploration probability
   - Final epsilon: Minimum exploration probability
   - Decay rate: How quickly to reduce exploration

Our implementation tests multiple configurations to find the optimal balance of these parameters.

## Policy Comparison (`policy_comparison.py`)

### MLP vs. CNN Policy

We compare two policy network architectures:

1. **MLP (Multi-Layer Perceptron)**:
   - Simpler architecture with fully connected layers
   - Works well for low-dimensional state spaces
   - Less effective for processing raw pixel data

2. **CNN (Convolutional Neural Network)**:
   - Specialized for processing spatial data like images
   - Can extract relevant features from raw pixels
   - More computationally intensive but better performance on visual tasks

For Atari games, CNN policies typically outperform MLP policies because they can better process the spatial information in game frames. Our comparison quantifies this difference in performance.

## Key Design Decisions

1. **Modular Implementation**: Separate scripts for different aspects of the project make it easier to experiment with different components.

2. **Consistent Evaluation**: Using the same metrics across experiments allows for fair comparisons.

3. **Logging and Visualization**: TensorBoard integration enables tracking of training progress and comparison between different runs.

4. **Hyperparameter Organization**: Structured hyperparameter sets make it easy to track which configurations were tested and their results.

5. **Reduced Training Time for Experiments**: Using fewer timesteps for hyperparameter tuning and policy comparison allows for quicker iteration while still providing meaningful results.

## Expected Results

- **CNN vs. MLP**: CNN policy should significantly outperform MLP for Atari games
- **Learning Rate**: Moderate values (around 1e-4) typically work best
- **Gamma**: Values close to 1 (0.99) are generally optimal for Atari games
- **Batch Size**: 32-64 typically provides a good balance of stability and speed
- **Exploration**: Starting high (1.0) and gradually decreasing to a low value (0.05) over 10% of training allows sufficient exploration