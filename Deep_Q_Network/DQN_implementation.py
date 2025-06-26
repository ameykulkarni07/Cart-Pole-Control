import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
LR = 1e-3
MEMORY_SIZE = 10000
EPISODES = 1000


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize networks
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)


# Training function
def train_dqn():
    epsilon = EPS_START
    rewards = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0

        for t in range(500):  # Max steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = torch.tensor([[env.action_space.sample()]], dtype=torch.long)
            else:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Store transition
            memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([done]))

            state = next_state
            episode_reward += reward

            # Train if enough samples
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                # Compute Q values
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                next_state_batch = torch.cat(batch.next_state)
                done_batch = torch.cat(batch.done)

                current_q = policy_net(state_batch).gather(1, action_batch)
                next_q = target_net(next_state_batch).max(1)[0].detach()
                expected_q = reward_batch + (GAMMA * next_q * (1 - done_batch.float()))

                # Compute loss
                loss = nn.MSELoss()(current_q.squeeze(), expected_q)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        rewards.append(episode_reward)

        print(f'Episode {episode}, Reward: {episode_reward:.1f}, Epsilon: {epsilon:.2f}')

    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Progress')
    plt.grid()
    plt.savefig('dqn_training_2.png')
    plt.show()


if __name__ == '__main__':
    train_dqn()
    torch.save(policy_net.state_dict(), 'dqn_cartpole_2.pth')