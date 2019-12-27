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


### Hyperparameters

## Plot of Rewards

## Ideas for Future Work