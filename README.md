# Cart-Pole-Control
A repository which tests various Reinforcement Learning algorithms to control the Cart Pole with inverted pendulum. The environment is from OpenAI Gymnasium. This repository was created when I started learning the practical implementation of reinforcement learning. 

https://github.com/user-attachments/assets/233083fa-76a2-41fb-8027-988ed9db6690

(This is the video output file from one of the controls done by DQN.)

## Problem Description
The CartPole-v1 environment from Gymnasium is a classic reinforcement learning benchmark where the agent must balance a pole on a moving cart. The system is controlled by applying forces of +1 (right) or -1 (left) to the cart. The episode ends when the pole tilts more than 12 degrees from vertical or the cart moves more than 2.4 units from the centre.

![image](https://github.com/user-attachments/assets/a30c0f8f-47a2-4cc1-b08a-9058c5fdf53e)

![image](https://github.com/user-attachments/assets/c5e3f767-1ac5-4692-a328-980053ffbd23)


## Implementations
- **Decision Transformer** - Offline RL approach that treats RL as a sequence modelling problem
  - Implemented with PyTorch
  - Includes training script and trained model in `/Decision Transformer` and evaluation scripts in `/Evaluation` 

Decision Transformer was trained on 1000 episodes and evaluated on several episodes. The result plots generated are for the longest episode achieved. The context length information is obtained from a simpler PID controller. 
The decision transformer was tested for its generalisation ability. Two models of decision transformers were trained with different training dataset. First training dataset consisit of a simple physics based controller and second one consist of PID controller (same as the one used for context length). It was oberved that the decision transformer trained using PID controller based decisions was able to control the cartpole for all the evaluation episodes. This was not the case, when the decision transformer trained on physics based controller was utilised to control the cart.

- **Deep Q Network** - Classical RL algorithm based on Q values. 
  - Implemented Experience Replay and target network for stable training
  - Includes training script and trained model in /Deep_Q_Network` and evaluation scripts in `/Evaluation`
 
The DQN model was trained on 1000 episodes and was evaluated on several episodes. The result plots and control video was saved.



