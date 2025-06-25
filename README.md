# Cart-Pole-Control
A repository which tests various Reinforcement Learning algorithms to control the Cart Pole with inverted pendulum. The environment is from OpenAI Gymnasium. This repository was created when I started learning the practical implementation of reinforcement learning. 

https://github.com/user-attachments/assets/3765bcfb-adba-426c-ac10-5fc6e37f4741

(This is the video output file from one of the controls done by Decision Transformer.)

## Problem Description
The CartPole-v1 environment from Gymnasium is a classic reinforcement learning benchmark where the agent must balance a pole on a moving cart. The system is controlled by applying forces of +1 (right) or -1 (left) to the cart. The episode ends when the pole tilts more than 12 degrees from vertical or the cart moves more than 2.4 units from the centre.

![image](https://github.com/user-attachments/assets/a30c0f8f-47a2-4cc1-b08a-9058c5fdf53e)

![image](https://github.com/user-attachments/assets/c5e3f767-1ac5-4692-a328-980053ffbd23)


## Implementations
- **Decision Transformer** - Offline RL approach that treats RL as a sequence modelling problem
  - Implemented with PyTorch
  - Includes training in `/Decision Transformer` and evaluation scripts in `/Evaluation` 
  - Pre-trained models available in `/Decision Transformer`

Decision Transformer was trained on 1000 episodes and evaluated on 500 episodes. The result plots generated are for the longest episode achieved. The context length information is obtained from a simpler PID controller.




