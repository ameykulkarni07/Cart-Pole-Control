import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# ----------------------
# Part 1: Decision Transformer
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



"""
Part 1 - defines the Decision Transformer Model. The action dimensions and state dimensions are fixed for the environment of catpole-v1. 
The model takes in two actions - 0 : Push the cart to the left and 1 : Push the cart to the right. 
The observation space is a $ dimensional array with [cart position, cart velocity, pole angle, pole angular velocity].
The embedding layers are used to assign a vector representation to the states, actions and return to go. 
The model consist of 'n' number of layers and each layer has 'm' number of heads. The attention mechanism (part of the heads) learns the contextual information \
from the input sequence of states, actions and returns to go.
"""

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=4, act_dim=2, embed_dim=64, num_layers=3, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Embedding(act_dim, embed_dim)
        self.return_embed = nn.Linear(1, embed_dim)
        self.time_embed = nn.Embedding(1000, embed_dim)

        # Transformer Archutecture
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        # Prediction heads
        # The action head predicts the next action based on the embedded states, actions and returns to go. This is called Teacher Forcing Method.
        self.action_head = nn.Linear(embed_dim, act_dim)

    def forward(self, states, actions, returns, timesteps):
        batch_size = states.shape[0]
        # Embeddings
        state_emb = self.state_embed(states)
        action_emb = self.action_embed(actions)
        # Handle different dimensions for returns. This is to easily use this same model for the evaluation purpose.
        if returns.dim() == 2:
            returns = returns.unsqueeze(-1)
        return_emb = self.return_embed(returns)
        time_emb = self.time_embed(timesteps)
        x = state_emb + action_emb + return_emb + time_emb
        x = self.transformer(x)
        action_logits = self.action_head(x)
        return action_logits

# ----------------------
# Part 2: Dataset Handling
# ----------------------
"""
In this part, we work on the trajectories which are already collected from the envcironment. The trajectories are stored in a pickle file. 
The trajectories are prepared to have 'k' as the sequence length. The dataset is prepared in such a way that each sample contains the states, \
actions, returns to go and timesteps.
We use 1000 episodes as generated from the generation script. The maximum length of the trajectory can be 500 steps. The context length is set to 50 steps.
"""


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, seq_len=60):
        self.seq_len = seq_len
        self.data = []

        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            returns = traj['returns_to_go']

            for i in range(len(states) - seq_len):
                self.data.append((
                    states[i:i + seq_len],
                    actions[i:i + seq_len],
                    returns[i:i + seq_len],
                    np.arange(i, i + seq_len)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states, actions, returns, timesteps = self.data[idx]
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(returns),
            torch.LongTensor(timesteps)
        )

# ----------------------
# Part 3: Training Loop
# ----------------------
"""
This section imports the training dataset, initialises the model and then trains the above defined model.
The batch size is important parameter. Lower is the batch size, better are the results. But, this also leads to longer training time.

"""

def train():
    # Load pre-collected dataset
    with open('cartpole_dt_dataset_iter2.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    # Prepare dataset
    dataset = TrajectoryDataset(trajectories)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    # Initialize model
    model = DecisionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device}...")
    # 10ß epochs of training
    for epoch in range(100):
        model.train()
        total_loss = 0

        for states, actions, returns, timesteps in dataloader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            returns = returns.to(device, non_blocking=True)
            timesteps = timesteps.to(device, non_blocking=True)

            # Forward pass. Important that we only show past information to the model.
            preds = model(
                states[:, :-1],  # States up to t-1
                actions[:, :-1],  # Actions up to t-1
                returns[:, :-1],  # Returns up to t-1
                timesteps[:, :-1]  # Timesteps up to t-1
            )
            # Calculate loss
            loss = criterion(
                preds.reshape(-1, 2),
                actions[:, 1:].reshape(-1)  # Predict next action
            )
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'dt_cartpole_seqlen60.pth')
    print("Training complete. Model saved")

# ----------------------
# Main Execution
# ----------------------

if __name__ == "__main__":
    train()
