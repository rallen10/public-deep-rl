from unityagents import UnityEnvironment
import numpy as np
from continuous_control_brain_agent import ActorCriticMind, train_actor_critic_mind

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils.utils as U
from utils.network_models import DeepNetwork, GaussianActorNetwork

# create environment
env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=True)

# create a learning agent
brain_name = env.brain_names[0]
vanilla_ppo_mind = ActorCriticMind(brain_name, env.brains[brain_name], policy_lr=2e-4, value_lr=1e-3)

# get number of agents
env_info = env.reset(train_mode=True)[brain_name]
n_agents = len(env_info.agents)

# run training
train_actor_critic_mind(env, vanilla_ppo_mind, n_agents,
    n_episodes = 500,
    save_rate = 1,
    n_epochs = 4,
    desired_batch_size = 64,
    desired_minibatch_size = 16,
    gamma = 0.9,
    lam = 0.1,
    prefix = "lambda010",
    beta_start = 0.1, beta_end = 0.0005, beta_decay = 0.99,
    epsilon_start=0.3, epsilon_end=0.1, epsilon_decay = 0.99,
    v_epsilon_start=0.3, v_epsilon_end=0.1, v_epsilon_decay=0.99)

# clean up
env.close()

