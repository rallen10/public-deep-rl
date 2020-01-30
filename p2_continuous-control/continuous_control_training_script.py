import argparse
from unityagents import UnityEnvironment
import numpy as np
from continuous_control_brain_agent import ActorCriticMind, train_actor_critic_mind

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils.utils as U
from utils.network_models import DeepNetwork, GaussianActorNetwork


def parse_args(args_feed):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--exp-name", type=str, default="default", help="name of training experiment")
    parser.add_argument("--policy-lr", type=float, default=2e-4, help="learning rate for policy network")
    parser.add_argument("--value-lr", type=float, default=1e-3, help="learning rate for value network")
    parser.add_argument("--gae-lambda", type=float, default=0.9, help="lambda value in generalized advantage estimate")
    return parser.parse_args(args_feed)

if __name__ == "__main__":
    arglist = parse_args(sys.argv[1:])

    # create environment
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=True)

    # create a learning agent ('mind')
    brain_name = env.brain_names[0]
    ppo_mind = ActorCriticMind(
        brain_name, 
        env.brains[brain_name], 
        policy_lr=arglist.policy_lr, 
        value_lr=arglist.value_lr)

    # get number of agents
    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)

    # run training
    train_actor_critic_mind(env, ppo_mind, n_agents,
        n_episodes = 200,
        save_rate = 1,
        n_epochs = 4,
        desired_batch_size = 64,
        desired_minibatch_size = 16,
        gamma = 0.9,
        lam = arglist.gae_lambda,
        prefix = arglist.exp_name,
        beta_start = 0.1, beta_end = 0.0005, beta_decay = 0.99,
        epsilon_start=0.3, epsilon_end=0.1, epsilon_decay = 0.99,
        v_epsilon_start=0.3, v_epsilon_end=0.1, v_epsilon_decay=0.99)

    # clean up
    env.close()


