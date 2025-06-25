import gym
import numpy as np
import pickle

def collect_trajectories(env, num_episodes=1000, max_episode_length=500):
    trajectories = []
    termination_stats = {'angle': 0, 'position': 0, 'time': 0}

    for _ in range(num_episodes):
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0

        for t in range(max_episode_length):
            cart_pos, cart_vel, pole_angle, pole_vel = state

            # Control logic with velocity compensation
            angle_threshold = 0.05  # ~2.87 degrees
            vel_compensation = 0.2 * pole_vel  # Dampen angular velocity
            action = 1 if (pole_angle + vel_compensation) > 0 else 0
            # Conservative position management
            pos_threshold = 1.8  # Close to but below 2.4 limit
            if abs(cart_pos) > pos_threshold:
                action = 0 if cart_pos > 0 else 1
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Track termination reasons
            if terminated:
                if abs(pole_angle) > 0.2095:
                    termination_stats['angle'] += 1
                elif abs(cart_pos) > 2.4:
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

        returns += [0] * (max_episode_length - len(returns))
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'returns_to_go': np.array(returns[:len(states)])
        })

    # Print some details for understanding
    avg_length = np.mean([len(t['states']) for t in trajectories])
    print("\n=== Data Collection Report ===")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Successful episodes (500 steps): {termination_stats['time']}")
    print(f"Angle failures: {termination_stats['angle']}")
    print(f"Position failures: {termination_stats['position']}")

    return trajectories

# Create and save dataset
env = gym.make("CartPole-v1")
trajectories = collect_trajectories(env)

with open('cartpole_dt_dataset.pkl', 'wb') as f:
    pickle.dump(trajectories, f)

all_states = np.concatenate([t['states'] for t in trajectories])
state_mean = np.mean(all_states, axis=0)
state_std = np.std(all_states, axis=0)
print("State mean:", state_mean)
print("State std:", state_std)