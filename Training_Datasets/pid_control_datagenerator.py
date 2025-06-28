import gym
import numpy as np
import pickle


def collect_trajectories_with_pid(env, num_episodes=1000, max_episode_length=500):
    trajectories = []
    termination_stats = {'angle': 0, 'position': 0, 'time': 0}

    for ep in range(num_episodes):
        state, _ = env.reset()
        states, actions, rewards = [], [], []

        # PID controller variables
        prev_error = 0
        integral = 0

        for t in range(max_episode_length):
            # Apply PID controller
            angle = state[2]  # Pole angle
            angle_tolerance = 0.02  # About 1 degree of tolerance
            if abs(angle) < angle_tolerance:
                error = 0  # No error if within tolerance
            else:
                error = angle  # Otherwise error is the angle (target is 0)
            integral += error
            derivative = error - prev_error
            prev_error = error

            # Calculate control signal
            control_signal = 1.2 * error + 0.08 * integral + 8.0 * derivative
            action = 1 if control_signal > 0 else 0

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Track termination reasons
            if terminated:
                if abs(angle) > 0.2095:
                    termination_stats['angle'] += 1
                elif abs(state[0]) > 2.4:
                    termination_stats['position'] += 1
            elif truncated:
                termination_stats['time'] += 1

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break

        # Calculate returns-to-go
        returns = []
        cumulative = 0
        for r in reversed(rewards):
            cumulative += r
            returns.insert(0, cumulative)

        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'returns_to_go': np.array(returns)
        })

        # Print progress
        if (ep + 1) % 100 == 0:
            print(f"Collected {ep + 1}/{num_episodes} episodes")

    # Print statistics
    avg_length = np.mean([len(t['states']) for t in trajectories])
    print("\n=== Data Collection Report ===")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Successful episodes (500 steps): {termination_stats['time']}")
    print(f"Angle failures: {termination_stats['angle']}")
    print(f"Position failures: {termination_stats['position']}")

    return trajectories


# Create environment and collect data
env = gym.make("CartPole-v1")
trajectories = collect_trajectories_with_pid(env)

# Calculate normalization statistics
all_states = np.concatenate([t['states'] for t in trajectories])
state_mean = np.mean(all_states, axis=0)
state_std = np.std(all_states, axis=0)
print("State mean:", state_mean)
print("State std:", state_std)

# Prevent division by zero
state_std = np.where(state_std < 1e-6, 1.0, state_std)

# Save dataset with normalization parameters
with open('cartpole_dt_dataset_pid.pkl', 'wb') as f:
    pickle.dump({
        'trajectories': trajectories,
        'state_mean': state_mean,
        'state_std': state_std
    }, f)

print("Dataset saved successfully with normalization parameters")