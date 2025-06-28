import matplotlib.pyplot as plt
import numpy as np
import pickle


def visualize_training_data(data_path='cartpole_dt_dataset_pid.pkl'):
    # Load the dataset
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        trajectories = data['trajectories']

    # Extract components from all trajectories
    all_states = np.concatenate([t['states'] for t in trajectories])
    all_actions = np.concatenate([t['actions'] for t in trajectories])
    all_returns = np.concatenate([t['returns_to_go'] for t in trajectories])
    episode_lengths = [len(t['states']) for t in trajectories]

    # Create visualization figure
    plt.figure(figsize=(18, 12))
    plt.suptitle('Training Data Statistics', y=1.02, fontsize=16)

    # 1. Action Distribution
    plt.subplot(2, 3, 1)
    plt.hist(all_actions, bins=np.arange(-0.5, 2.5), edgecolor='black', rwidth=0.8)
    plt.xticks([0, 1], ['Left (0)', 'Right (1)'])
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)

    # 2. State Components Distribution
    plt.subplot(2, 3, 2)
    state_labels = ['Cart Position', 'Cart Velocity',
                    'Pole Angle', 'Pole Velocity']
    for i in range(4):
        plt.hist(all_states[:, i], bins=50, alpha=0.5, label=state_labels[i])
    plt.title('State Variable Distributions')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. Returns-to-Go Distribution
    plt.subplot(2, 3, 3)
    plt.hist(all_returns, bins=50, edgecolor='black')
    plt.title('Returns-to-Go Distribution')
    plt.xlabel('Return-to-Go Value')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)

    # 4. Episode Length Analysis
    plt.subplot(2, 3, 4)
    plt.hist(episode_lengths, bins=np.arange(0, 550, 50), edgecolor='black')
    plt.title('Episode Length Distribution')
    plt.xlabel('Episode Length (timesteps)')
    plt.ylabel('Count')
    plt.xticks(np.arange(0, 550, 100))
    plt.grid(alpha=0.3)

    # 5. Temporal Return Patterns
    plt.subplot(2, 3, 5)
    max_length = max(episode_lengths)
    returns_matrix = np.full((len(trajectories), max_length), np.nan)

    for i, traj in enumerate(trajectories):
        length = len(traj['returns_to_go'])
        returns_matrix[i, :length] = traj['returns_to_go']

    mean_returns = np.nanmean(returns_matrix, axis=0)
    plt.plot(mean_returns, linewidth=2)
    plt.title('Average Returns-to-Go vs Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Return-to-Go')
    plt.grid(alpha=0.3)

    # 6. Example Trajectory Visualization
    plt.subplot(2, 3, 6)
    longest_idx = np.argmax(episode_lengths)
    example_traj = trajectories[longest_idx]['states']

    for i in range(4):
        plt.plot(example_traj[:, i], label=state_labels[i])

    plt.title(f'Longest Trajectory (Length: {episode_lengths[longest_idx]} steps)')
    plt.xlabel('Timestep')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# Run the visualization
visualize_training_data()