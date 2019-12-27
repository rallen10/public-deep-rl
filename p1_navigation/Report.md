# Navigation Report

This work uses Q-learning with a deep neural network function approximator to train an autonomous agent to navigate through an environment containing positive and negative rewards.  
The environment consists of a square, 2D world randomly populated with bananas and is rendered using the [Unity game engine](https://unity.com/) and is derived from [Unity's ML-Agent library](https://github.com/Unity-Technologies/ml-agents). Yellow bananas are positive rewards that the agent should collect, and blue bananas are negative rewards that the agent should avoid
The combination of deep neural networks and Q-learning is often referred to as [Deep Q-Networks (DQN)](https://www.nature.com/articles/nature14236?wm=book_wap_0005) and was shown to be highly effective at learning human-level skills at a range of tasks.

## Learning Algorithm

### Deep Q-Networks

### Double Deep Q-Networks

### Model Architecture

### Code Structure

The code is broken into jupyter notebooks (`.ipynb`), python scripts and modules (`.py`), markdown files (`.md`), model checkpoints (`.pth`), and data log files (`.pkl`). 

The jupyter notebook `Navigation.ipynb` is effectively the "main" function for the code. It is responsible for establishing the environment (i.e. `unityagents.UnityEnvironment`), creating the agents that interact with the environment (i.e. `dqn_brain_agent.BrainAgent`), and executing the training for various agents (i.e. `dqn_brain_agent.train_dqn`). 

The python module `dqn_brain_agent.py` contains most of the source code run by `Navigation.ipynb`. The `dqn_brain_agent` module provides a class for defining trainable agents (`BrainAgent`) and a function for stepping through the training processes (`train_dqn`). 
The `BrainAgent` class defines objects for storing and training the agents that interact with the environment. The policy for selecting actions given an observation of the environment is an [epsilon-greedy strategy](https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies) based on the estimated action value function and is defined in `BrainAgent.act`. The action value function is stored in `BrainAgent.dqn_local`, is a object of type `DeepQNetwork` defined in the `model.py` module. The `BrainAgent.dqn_local` action value function is a deep neural network defined using PyTorch. The network takes input vectors of size 37 (i.e. the observation space size for the environment) and outputs vectors of size 4 (i.e. the action space size for the environment). 

### Hyperparameters

## Plot of Rewards

## Ideas for Future Work