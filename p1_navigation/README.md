[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

(Gif Credit: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Create and activate a conda environment to establish the correct dependencies (e.g. Python3, PyTorch, etc)

```
conda-env create -f ../environment.yml 		# this only needs to be run once
source activate public_drlnd 				# this needs to be run everytime
```

2. Download the Banana environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in the `public-deep-rl` GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

## Instructions

### Training

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### Analysis

The `training_analysis.py` module contains functions for plotting learning curves and visualizing the behavior of a trained agent.

To plot learning curves with data saved in `.pkl` files, you can run the following from the command line (this example assumes there are pickle files titled `vanilla_scores.pkl` and `ddqn_scores.pkl`:

```
python training_analysis.py vanilla_scores.pkl ddqn_scores.pkl
```

To visualize the behavior of a trained agent, you must use a python interactive shell like `ipython`. Assuming that an agent has been trained and the model saved in `vanilla_checkpoint.pth`, you can run the following to visualize the learned behavior

```
ipython
from training_analysis.py import visualize_agent
visualize_agent('vanilla_checkpoint.pth') 
```