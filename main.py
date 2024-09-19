import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        self.done = False

    def reset(self):
        self.state = (0, 0)
        self.done = False
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # left
            y = max(0, y - 1)

        self.state = (x, y)
        reward = -1
        if self.state == self.goal:
            reward = 100
            self.done = True
        return self.state, reward, self.done


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q


def train(env, agent, episodes):
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        if episode % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {episode}")
            plot_training_progress(rewards_per_episode)
            plot_q_table(agent.q_table)
            sleep(0.1)

    return rewards_per_episode


def plot_training_progress(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def plot_q_table(q_table):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    actions = ['Up', 'Right', 'Down', 'Left']
    
    for i, ax in enumerate(axes):
        im = ax.imshow(q_table[:,:,i], cmap='hot')
        ax.set_title(f'Q-values for {actions[i]}')
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


env = GridWorld(size=5)
agent = QLearningAgent(state_size=5, action_size=4)
rewards = train(env, agent, episodes=1000)


plot_training_progress(rewards)
plot_q_table(agent.q_table)

print("Training completed!")
