# Learning to Play Doom from Demonstrations

This code implements the DQfD algorithm presented in [Deep Q-learning from Demonstrations (DQfD)](https://arxiv.org/abs/1704.03732) to solve Doom environments. Demonstrations have to be recorded first using `record_demonstrations.py` if one wants to use the DQfD algorithm in `train.py`. `train.py` contains the main training and optimization loop, `models.py` contains the PyTorch neural network and `utils.py` contains some utility classes for processing frames, storing transitions, and an epsilon-greedy policy.

Dependencies:
- PyTorch
- Open AI Gym
- [Gym-Doom](https://github.com/ppaquette/gym-doom)
- tensorboard
- [tensorboard logger](https://github.com/TeamHG-Memex/tensorboard_logger)

The tensorboard dependencies can be removed by commenting out their corresponding lines of code in `train.py`