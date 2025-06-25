import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from Decision_Transformer.Decision_transformer import DecisionTransformer
import time
from gym import wrappers
#import moviepy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleEvaluator:
    def __init__(self, model_path, seq_len):
        self.env = gym.make("CartPole-v1")
        self.model = DecisionTransformer().to(device)
        self.seq_len = seq_len

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Dataset normalization values (adjust based on your dataset)
        self.state_mean = torch.FloatTensor([0.0013, 0.0028, 0.00003, 0.00001]).to(device)
        self.state_std = torch.FloatTensor([0.5603, 0.1746, 0.0099, 0.2160]).to(device)

        # PID controller parameters (tuned for CartPole)
        self.kp = 1.2  # Proportional gain
        self.ki = 0.08  # Integral gain
        self.kd = 8.0  # Derivative gain
        self.prev_error = 0
        self.integral = 0

    def _prepare_inputs(self, history):
        states = torch.FloatTensor(np.array(history['states'][-self.seq_len:])).to(device)
        actions = torch.LongTensor(history['actions'][-self.seq_len:]).to(device)
        returns = torch.FloatTensor(history['returns'][-self.seq_len:]).to(device)
        timesteps = torch.LongTensor(history['timesteps'][-self.seq_len:]).to(device)

        states = (states - self.state_mean) / self.state_std
        returns = returns.unsqueeze(-1) / 100.0

        return (states.unsqueeze(0),
                actions.unsqueeze(0),
                returns.unsqueeze(0),
                timesteps.unsqueeze(0))

    def _pid_controller(self, state):
        """Simple PID controller for balancing the pole during warm-up phase."""
        angle = state[2]  # Pole angle (radians)
        angular_velocity = state[3]  # Pole angular velocity

        error = angle  # The goal is to keep the pole at 0 radians
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        # PID control signal
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Convert PID output to a discrete action (0 = Left, 1 = Right)
        action = 1 if control_signal > 0 else 0
        return action

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
            history = {
                'states': [state],
                'actions': [0],
                'returns': [500],
                'timesteps': [0]
            }

            current_ep = {
                'states': [], 'actions': [], 'timesteps': [],
                'angles': [], 'cart_velocities': [], 'cart_positions': []
            }

            for t in range(500):
                if t < self.seq_len:
                    action = self._pid_controller(state)
                else:
                    states, actions, returns, timesteps = self._prepare_inputs(history)
                    with torch.no_grad():
                        logits = self.model(states, actions, returns, timesteps)
                        action = torch.argmax(logits[0, -1]).item()

                next_state, reward, done, _, _ = self.env.step(action)

                history['states'].append(next_state)
                history['actions'].append(action)
                history['returns'].append(history['returns'][-1] - reward)
                history['timesteps'].append(t + 1)

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
            results['rewards'].append(duration)

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

            print(f"Episode {ep + 1:3d}/{num_episodes} | Steps: {duration:4d}")

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
        plt.xlabel("Episode")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("cartpole_evaluation_results.png")
        plt.show()

    '''
    # Uncommeent the section to animate and save the longest episode

    def _animate_longest_episode(self, results):
        """Saves a video of the longest episode."""
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
            name_prefix="longest_episode",
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
        '''

    def _print_statistics(self, results):
        durations = np.array(results['durations'])
        print("\n=== Evaluation Results ===")
        print(f"Longest Episode: {len(results['longest_episode']['timesteps'])} steps")
        print(f"Average Duration: {durations.mean():.1f} ± {durations.std():.1f}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_location = os.path.join(script_dir, '..', 'Decision_Transformer', 'dt_cartpole_iter2_seqlen50.pth')
    model_location = os.path.normpath(model_location)
    evaluator = CartPoleEvaluator(model_path=model_location, seq_len=50)
    results = evaluator.evaluate(num_episodes=500)