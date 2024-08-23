# Deep Learning Grid World Q-Learning

## Overview

This repository contains an implementation of a Q-learning algorithm to solve a grid world environment using deep learning techniques. The environment consists of a 5x5 grid with obstacles, rewards, and a goal state. The agent learns to navigate this grid to maximize its cumulative reward using Q-learning.

## Files

- `deep_learning_grid_world_q_learning.py`: Contains the main implementation of the Q-learning algorithm, including:
  - `create_grid_world(ax)`: Function to create and visualize the grid world.
  - `epsilon_greedy(Q, state, epsilon)`: Function to select an action using the epsilon-greedy policy.
  - `step(Q, state, action, alpha, gamma)`: Function to perform a step in the environment and update Q-values.
  - `q_learning_agent(alpha_values, num_episodes)`: Function to train the Q-learning agent with different alpha values.
  - `visualize_q_values(Q)`: Function to visualize the learned Q-values.

## Usage

1. **Run the Deep Learning Q-learning Agent**

   Execute the script to train the Q-learning agent with different learning rates (`alpha_values`). The training process includes visualization of the agent's movement in the grid world and updates to the Q-values.

    python deep_learning_grid_world_q_learning.py

2. **Visualize Q-values**

   After training, the Q-values are visualized to show the learned state values.

## Explanation

### Grid World

The grid world consists of a 5x5 grid with:
- **Obstacles**: Cells that are blocked and cannot be traversed.
- **Rewards**: Cells that provide rewards (+5 or +10).
- **Goal**: The cell at (4, 4) where the agent receives a reward of +10 and the episode terminates.

### Deep Q-learning Algorithm

- **Epsilon-Greedy Policy**: Balances exploration and exploitation.
- **Learning Rate (Alpha)**: Controls the rate at which the Q-values are updated.
- **Discount Factor (Gamma)**: Determines the importance of future rewards.

### Visualization

- **Grid World**: The grid is displayed with obstacles, rewards, and the agent's path.
- **Q-values**: Visualized as a heatmap to show the learned state values.

### Screenshots

<img width="959" alt="ary1" src="https://github.com/user-attachments/assets/6d36f435-46f1-45e9-8215-321e7c8f54f6">

### Output Video

https://github.com/user-attachments/assets/5a1f35c8-bd06-43cc-97d9-961f69286a54

## Notes

- The script includes an early stopping condition if the average reward exceeds a threshold over a window of episodes.
- The agent's progress is visualized in real-time during training.

## Contributing

- Tashfeen Abbasi
- [Laiba Mazhar](https://github.com/laiba-mazhar)

Feel free to fork the repository and submit pull requests. For issues or feature requests, please open an issue on GitHub.

