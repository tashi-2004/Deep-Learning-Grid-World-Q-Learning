import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

#`````````````````````````````````````````````````````````````` GRID ````````````````````````````````````````````````````````````
def create_grid_world(ax):
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xticklabels(np.arange(1, 7, 1), fontsize=10)
    ax.set_yticklabels(np.arange(1, 7, 1), fontsize=10)
    ax.grid(True)
    ax.tick_params(axis='x', which='both', pad=10)
    ax.tick_params(axis='y', which='both', pad=10)

    ax.add_patch(patches.Rectangle((4, 4), 1, 1, fill=True, color='cyan'))
    ax.text(4.5, 4.5, '+10', ha='center', va='center', fontsize=12, color='black')

    ax.add_patch(patches.Rectangle((3, 1), 1, 1, fill=True, color='cyan'))
    ax.text(3.5, 1.5, '+5', ha='center', va='center', fontsize=12, color='black')
    ax.arrow(3.5, 1.5, 0, 1.8, head_width=0.2, head_length=0.2, fc='blue', ec='blue', lw=2, zorder=20)

    obstacles = [(2, 2), (3, 2), (4, 2), (2, 3)]
    for (x, y) in obstacles:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=True, color='black'))

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    plt.gca().invert_yaxis()

#``````````````````````````````````````````````````````````````` EPSILON `````````````````````````````````````````````````````````````
def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3])
    else:
        return np.argmax(Q[state])

#```````````````````````````````````````````````````````````````` STEP ```````````````````````````````````````````````````````````````
def step(Q, state, action, alpha, gamma):
    x, y = state
    reward = -1
    new_x, new_y = x, y

    if action == 0: new_y -= 1
    elif action == 1: new_y += 1
    elif action == 2: new_x -= 1
    elif action == 3: new_x += 1

    new_x = max(0, min(4, new_x))
    new_y = max(0, min(4, new_y))

    if (new_x, new_y) in [(2, 2), (3, 2), (4, 2), (2, 3)]:
        return state, reward - 1

    if (new_x, new_y) == (3, 1):
        new_x, new_y = 3, 3
        reward += 5

    if (new_x, new_y) == (4, 4):
        reward += 10
        Q[(new_x, new_y)] = [0, 0, 0, 0]

    if (abs(4 - new_x) + abs(4 - new_y)) < (abs(4 - x) + abs(4 - y)):
        reward += 0.5

    next_state = (new_x, new_y)
    max_next_Q = max(Q[next_state])
    Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])

    return next_state, reward


#```````````````````````````````````````````````````````````````` LEARNING AGENT  `````````````````````````````````````````````````````````````
def q_learning_agent(alpha_values, num_episodes=100):
    global Q
    for alpha in alpha_values:
        print("\t\t\t\t\t_______________________________")
        print("\t\t\t\t\t| Training, Episodes, Rewards |")
        print("\t\t\t\t\t|_____________________________|")
        print("\n")
        print(f"Training with learning rate alpha = {alpha}")
        print("\n")
        fig, ax = plt.subplots()
        create_grid_world(ax)
        x, y = 1, 0
        state = (x, y)

        Q = {(i, j): [0, 0, 0, 0] for i in range(5) for j in range(5)}

        epsilon = 0.8
        gamma = 0.95
        max_steps_per_episode = 50 

        rewards_window = []

        for episode in range(num_episodes):
            state = (1, 0)
            total_reward = 0

            for step_num in range(max_steps_per_episode):
                action = epsilon_greedy(Q, state, epsilon)
                next_state, reward = step(Q, state, action, alpha, gamma)

                total_reward += reward
                state = next_state

                ax.clear()
                create_grid_world(ax)
                ax.add_patch(patches.Circle((state[0] + 0.5, state[1] + 0.5), 0.2, color='red', zorder=10))
                plt.draw()
                plt.pause(0.001)

                if state == (4, 4):
                    break


            rewards_window.append(total_reward)
            if len(rewards_window) > 30:
                rewards_window.pop(0)
            
            

            if len(rewards_window) == 30 and np.mean(rewards_window) > 10:
                print(f"Early stopping at episode {episode + 1} with average reward {np.mean(rewards_window)}") # Rarely Happens
                break

            epsilon = max(0.01, epsilon * 0.995)

            if episode % 10 == 0:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        plt.ioff()
        plt.show()


#``````````````````````````````````````````````````````````````` VISUALIZATION`````````````````````````````````````````````````````````````
def visualize_q_values(Q):
    state_values = np.zeros((5, 5))

    for (x, y), values in Q.items():
        state_values[y, x] = max(values)

    plt.figure(figsize=(5, 5))
    plt.imshow(state_values, cmap='plasma', interpolation='nearest')

    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{state_values[i, j]:.2f}", ha='center', va='center', color='black')

    plt.colorbar(label='State Value (Max Q-value)')
    plt.title('State Values in the Grid World')
    plt.show()


#```````````````````````````````````````````````````````````````````````` MAIN `````````````````````````````````````````````````````````````
if __name__ == '__main__':
    alpha_values = [1.0,0.5,0.1]
    q_learning_agent(alpha_values)
    visualize_q_values(Q)
