# Cart-Pole-Control
A repository which tests various Reinforcement Learning algorithms to control the Cart Pole with inverted pendulum. The environment is from OpenAI Gymnasium. This repository was created when I started learning the practical implementation of reinforcement learning. 

https://github.com/user-attachments/assets/3765bcfb-adba-426c-ac10-5fc6e37f4741

(This is the video output file from one of the controls done by Decision Transformer.)

## Implementations
- **Decision Transformer** - Offline RL approach that treats RL as a sequence modelling problem
  - Implemented with PyTorch
  - Includes training in `/Decision Transformer` and evaluation scripts in `/Evaluation` 
  - Pre-trained models available in `/Decision Transformer`

Decision Transformer was trained on 1000 episodes and evaluated on 500 episodes. The trained model was evaluated on 500 episodes. The result plots generated are for the longest episode achieved. The context length information is obtained from a simpler PID controller.
![cartpole_evaluation_results](https://github.com/user-attachments/assets/8a018d0d-d616-4d48-83d8-ad1097fbf0e0)



