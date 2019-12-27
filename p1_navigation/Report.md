[//]: # (Image References)

[image1]: learning_curves.png "DQN Learning Curves"

# Report: Navigation Project

This work uses Q-learning with a deep neural network function approximator to train an autonomous agent to navigate through an environment containing positive and negative rewards.  
The environment consists of a square, 2D world randomly populated with bananas and is rendered using the [Unity game engine](https://unity.com/) and is derived from [Unity's ML-Agent library](https://github.com/Unity-Technologies/ml-agents). Yellow bananas are positive rewards that the agent should collect, and blue bananas are negative rewards that the agent should avoid
The combination of deep neural networks and Q-learning is often referred to as [Deep Q-Networks (DQN)](https://www.nature.com/articles/nature14236?wm=book_wap_0005) and was shown to be highly effective at learning human-level skills at a range of tasks.

## Learning Algorithm

The code is broken into jupyter notebooks (`.ipynb`), python scripts and modules (`.py`), markdown files (`.md`), model checkpoints (`.pth`), and data log files (`.pkl`). 

The jupyter notebook `Navigation.ipynb` is effectively the "main" function for the code. It is responsible for establishing the environment (i.e. `unityagents.UnityEnvironment`), creating the agents that interact with the environment (i.e. `dqn_brain_agent.BrainAgent`), and executing the training for various agents (i.e. `dqn_brain_agent.train_dqn`). 

The python module `dqn_brain_agent.py` contains most of the source code run by `Navigation.ipynb`. The `dqn_brain_agent` module provides a class for defining trainable agents (`BrainAgent`) and a function for stepping through the training processes (`train_dqn`) as well as a class for creating an experience replay buffer (`ReplayBuffer`). 

The `BrainAgent` class defines objects for storing and training the agents that interact with the environment. The policy for selecting actions given an observation of the environment is an [epsilon-greedy strategy](https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies) based on the estimated action value function and is defined in `BrainAgent.act`. The action value function is stored in `BrainAgent.dqn_local`, is a object of type `DeepQNetwork` defined in the `model.py` module. The `BrainAgent.dqn_local` action value function is a deep neural network defined using PyTorch. The network takes input vectors of size 37 (i.e. the observation space size for the environment) and outputs vectors of size 4 (i.e. the action space size for the environment). In this work we use a deep neural network with fully-connected hidden layers of size 128, 64, and 32, respectively. Each layer uses a ReLU activation function.

As the `train_dqn` function steps the environment and agent through time, the agent must process and learn from experiences. The processing of experiences is managed by the `BrainAgent.process_step` method which stores new experiences in the agent's `memory` (which is an object of type `ReplayBuffer`), samples randomized past experiences for learning, and calls the `BrainAgent.learn` method. While all of the code is necessary for running the DQN learning process, it can be argued `BrainAgent.learn` is where the DQN algorithm is really "defined". In this method fixed Q-targets are computed using a separate `DeepQNetwork` stored in `BrainAgent.dqn_target` and these q-targets are used to compute the TD error relative to the action value function and subsequently the loss value to be minimized. 
`BrainAgent.learn` also defines a [Double Deep Q-Network (DDQN)](https://arxiv.org/abs/1509.06461) learning algorithm. This algorithm is similar to the "vanilla" DQN algorithm except that the action that maximizes the Q-function is estimated using one DQN (`BrainAgent.dqn_local`) and the value of that action is estimated using a different DQN (`BrainAgent.dqn_target`). This is meant to help prevent overestimation of action values because both networks must "agree" on the maximizing action.

Results from training are stored in `.pth` and `.pkl` files. The `.pth` are based on PyTorch's [`save`](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save) function and stored the model parameters for a trained neural network. The `.pkl` files are [pickle files](https://docs.python.org/3/library/pickle.html) that store the scores of different learning algorithms throughout their training process.

The `training_analysis.py` module defines functions for plotting and visualizing the training results. This module is designed so that it can be used in the `Navigation.ipynb` notebook as well as from command line. For example:

```
python training_analysis vanilla_scores.pkl ddqn_scores.pkl
```

will render a plot of the learning curves for the vanilla and double DQN algorithm based on training data stored in the respective `.pkl` pickle files. The ability to render plots from command line using saved pickle files makes it easier and faster to iterate on the post-processing scripts without re-running training.

The following hyperparameters were used during training:

+ BUFFER_SIZE = int(1e5)  # replay buffer size
+ BATCH_SIZE = 64         # minibatch size
+ GAMMA = 0.99            # discount factor
+ TAU = 1e-3              # for soft update of target parameters
+ LR = 5e-4               # learning rate 

## Plot of Rewards

![DQN Learning Curves][image1]

## Ideas for Future Work