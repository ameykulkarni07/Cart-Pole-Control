import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from gym import wrappers
from Deep_Q_Network.DQN_implementation import DQN
import moviepy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleDQNEvaluator:
    def __init__(self, model_path):
        self.env = gym.make("CartPole-v1")

        # Load the DQN model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        self.model = DQN(state_size=4,action_size=2).to(device)
        model_state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()


    def evaluate(self, num_episodes=500):
        results = {
            'durations': [],
            'rewards': [],
            'longest_episode': {
                'initial_state': None,
                'states': [], 'actions': [], 'timesteps': [],
                'angles': [], 'cart_velocities': [], 'cart_positions': []
            }
        }

        max_duration = 0
        longest_ep_data = None

        for ep in range(num_episodes):
            initial_state, _ = self.env.reset()
            state = initial_state

            current_ep = {
                'states': [], 'actions': [], 'timesteps': [],
                'angles': [], 'cart_velocities': [], 'cart_positions': []
            }

            total_reward = 0
            for t in range(500):
                # Convert state to tensor for DQN input
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                # Get action from DQN
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                    action = torch.argmax(q_values).item()

                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                current_ep['states'].append(next_state)
                current_ep['actions'].append(action)
                current_ep['timesteps'].append(t)
                current_ep['angles'].append(next_state[2])
                current_ep['cart_velocities'].append(next_state[1])
                current_ep['cart_positions'].append(next_state[0])

                state = next_state

                if done:
                    break

            duration = t + 1
            results['durations'].append(duration)
            results['rewards'].append(total_reward)

            if duration > max_duration:
                max_duration = duration
                longest_ep_data = {
                    'initial_state': initial_state,
                    'states': current_ep['states'].copy(),
                    'actions': current_ep['actions'].copy(),
                    'timesteps': current_ep['timesteps'].copy(),
                    'angles': current_ep['angles'].copy(),
                    'cart_velocities': current_ep['cart_velocities'].copy(),
                    'cart_positions': current_ep['cart_positions'].copy()
                }

            print(f"Episode {ep + 1:3d}/{num_episodes} | Steps: {duration:4d} | Reward: {total_reward:.1f}")

        if longest_ep_data:
            results['longest_episode'] = longest_ep_data

        self.env.close()
        self._generate_visualizations(results)
        self._animate_longest_episode(results)
        self._print_statistics(results)
        return results

    def _generate_visualizations(self, results):
        plt.figure(figsize=(15, 10))

        # Longest Episode Actions
        plt.subplot(2, 2, 1)
        plt.step(results['longest_episode']['timesteps'],
                 results['longest_episode']['actions'],
                 where='post')
        plt.title(f"Longest Episode ({len(results['longest_episode']['timesteps'])} steps)")
        plt.xlabel("Timestep")
        plt.ylabel("Action")
        plt.yticks([0, 1], ['Left', 'Right'])
        plt.grid(True)

        # Pole Angle Dynamics
        plt.subplot(2, 2, 2)
        plt.plot(results['longest_episode']['timesteps'],
                 results['longest_episode']['angles'])
        plt.title("Pole Angle Dynamics")
        plt.xlabel("Timestep")
        plt.ylabel("Angle (radians)")
        plt.grid(True)

        # Cart Position Analysis
        plt.subplot(2, 2, 3)
        plt.plot(results['longest_episode']['timesteps'],
                 results['longest_episode']['cart_positions'],
                 color='green')
        plt.title("Cart Position Dynamics")
        plt.xlabel("Timestep")
        plt.ylabel("Cart Position")
        plt.grid(True)
        plt.axhline(y=2.4, color='r', linestyle='--', label='Position Threshold')
        plt.axhline(y=-2.4, color='r', linestyle='--')
        plt.legend()

        # All Episode Durations
        plt.subplot(2, 2, 4)
        plt.hist(results['durations'], alpha=0.7)
        plt.title("All Episode Durations")
        plt.xlabel("Duration")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("cartpole_dqn_evaluation_results_2.png")
        plt.show()

    def _animate_longest_episode(self, results):
        longest_ep = results.get('longest_episode')
        if not longest_ep or not longest_ep.get('actions'):
            print("\nNo data available to save a video of the longest episode.")
            return

        video_path = "videos"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        print(f"\nSaving video of the longest episode to '{video_path}'...")

        # Create a new environment for rendering in the background
        anim_env = gym.make("CartPole-v1", render_mode="rgb_array")
        anim_env = wrappers.RecordVideo(
            anim_env,
            video_folder=video_path,
            name_prefix="dqn_longest_episode_model2",
            episode_trigger=lambda x: x == 0  # Record the first episode
        )

        # Reset the environment to start the episode and recording
        anim_env.reset()
        if longest_ep.get('initial_state') is not None:
            # Set the environment to the initial state of the longest episode
            anim_env.unwrapped.state = longest_ep['initial_state']
        for action in longest_ep['actions']:
            anim_env.step(action)

        # The video is saved when the environment is closed
        anim_env.close()
        print(f"Video saved successfully in the '{video_path}' directory.")


    def _print_statistics(self, results):
        durations = np.array(results['durations'])
        rewards = np.array(results['rewards'])
        print("\n=== Evaluation Results ===")
        print(f"Longest Episode: {len(results['longest_episode']['timesteps'])} steps")
        print(f"Average Duration: {durations.mean():.1f} ± {durations.std():.1f}")
        print(f"Average Reward: {rewards.mean():.1f} ± {rewards.std():.1f}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_location = os.path.join(script_dir, '..', '..', 'Deep_Q_Network', 'dqn_cartpole_2.pth')
    model_location = os.path.normpath(model_location)
    evaluator = CartPoleDQNEvaluator(model_path=model_location)
    results = evaluator.evaluate(num_episodes=500)