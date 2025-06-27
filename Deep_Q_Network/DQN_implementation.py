import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

# Hyperparameter
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
LR = 1e-3
MEMORY_SIZE = 10000
EPISODES = 1000

# ----------------------
# Part 1: DQN structure Definition
# ----------------------
"""
# Neural Network for DQN
The DQN is used to approximate the Q-value function. We use it instead of Tablular-Q-learning method as the state space is continous.
The neural network takes in the states as the input and the output is the Q values for the each action.
The neural network has two hidden layers with the 64 neurons each and the output layer has two neurons as we have two actions in the CartPole environment.
The activation function used is ReLU for the hidden layers and no activation function for the output layer.
The following function provides the architecture of the DQN.
"""
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

# ----------------------
# Part 2: Experience Replay Memory Implementation
# ----------------------
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory:
    # A fixed sized buffer is used to store the transitions. The transition contains the above shown constituents. It has state, action, next_state, reward and done.
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
     # The push function is used to add the transition to the memory.
    def push(self, *args):
        self.memory.append(Transition(*args))
     # The sample function is used to randomly sample the transitions from the memory. The need of this is to break the correlation. Explained in the documentation.
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# ----------------------
# Part 3: Environment Setup and Initialisation of the DQN networks. Also the Optimizer and Replay Memory
# ----------------------
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

# ----------------------
# Part 4: Training the DQN
# ----------------------
def train_dqn():
    """
    The epilson is the probability of taking a random action. It is used for exploration. The epsilon starts from EPS_START and decays to EPS_END.
    The reward list is used to store the rewards (time fir which the cart is balanced) for each episode.
    """
    epsilon = EPS_START
    rewards = []

    """
    Here the environment is reset for every episode and the initial state is obtained.
    For every epsode, the state is converted to a tensor and reshaped to match the input shape of the DQN.
    The episode_reward is initialized to 0 to keep track of the total reward for the episode.
    """
    for episode in range(EPISODES):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0

        """
        This section shows the logic for action selection. It shows the exploration and exploitation tradeoff. The specific action is selected based on the epsilon-greedy policy.
        The epilson number starts from EPS_START and decays to EPS_END. If the epsilon is greater than a random number, a random action is selected. 
        Otherwise, the action is selected based on the policy network. The policy network is used to predict the Q values for the current state and the action with maximum Q value is selected.
        """
        for t in range(500):
            if random.random() < epsilon:
                action = torch.tensor([[env.action_space.sample()]], dtype=torch.long)
            else:
                with torch.no_grad():
                    # "policy_net(state)" returns the Q values for the current state. "max(1)[1]" returns the index of the maximum Q value.
                    action = policy_net(state).max(1)[1].view(1, 1)

            # Execute action and get the next state, reward, and done flag The tensor is reshaped to match the input shape of the DQN.
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Store transition information in the replay memory
            memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([done]))
            state = next_state
            episode_reward += reward

            """
            The following code starts the training process. It only starts when the memory has enough transitions to sample a batch.
            The following code shows the training process of the DQN. It also shows the use of experience replay memory and target network which is used for stable training process
            The experience replay memory is used to store the transitions and sample them randomly to break the correlation between the transitions.
            The target network is used to compute the Q values for the next state. The target network is updated every TARGET_UPDATE episodes.
            More information can be found in the documentation.
            """
            if len(memory) >= BATCH_SIZE:
                # Sample a random batch of transitions from the replay memory
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                # The batch is segregated into individual tensors for state, action, reward, next_state and done.
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                next_state_batch = torch.cat(batch.next_state)
                done_batch = torch.cat(batch.done)

                """
                The current Q values are computed using the policy network. The action taken is used to gather the Q values for the specific action.
                The next Q values are computed using the target network. Target network is updated every TARGET_UPDATE episodes.  
                the detach function prevents the gradient computation for the target network. 
                This is done to achieve gradual updates and stable training.
                The expected Q values are computed using the Bellman equation. The reward is added to the discounted next Q values.
                """
                current_q = policy_net(state_batch).gather(1, action_batch)
                next_q = target_net(next_state_batch).max(1)[0].detach()
                expected_q = reward_batch + (GAMMA * next_q * (1 - done_batch.float()))

                # Compute loss
                loss = nn.MSELoss()(current_q.squeeze(), expected_q)
                # Optimize and update the policy network weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # Decay epsilon. we can control the rate of exploration by decaying the epsilon value.
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        rewards.append(episode_reward)
        print(f'Episode {episode}, Reward: {episode_reward:.1f}, Epsilon: {epsilon:.2f}')

    # Plot
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