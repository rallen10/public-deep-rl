import argparse
from unityagents import UnityEnvironment
import numpy as np
from actor_critic_mind import ActorCriticMind
from trainer import train_actor_critic_mind
import torch

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def parse_args(args_feed):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--n-episodes", type=int, default=10, help="training epochs per batch")
    parser.add_argument("--policy-filename", type=str, default="default_policy_checkpoint.pth", help="training checkpoint file for policy")
    return parser.parse_args(args_feed)

if __name__ == "__main__":
    arglist = parse_args(sys.argv[1:])

    # create environment
    env = UnityEnvironment(file_name='./Tennis_Linux/Tennis.x86_64', no_graphics=False)

    # load pre-trained brain
    brain_name = env.brain_names[0]
    ppo_mind = ActorCriticMind(
        brain_name, 
        env.brains[brain_name])
    ppo_mind.policy_network.load_state_dict(torch.load(arglist.policy_filename))

    # get number of agents
    train_mode = False
    env_info = env.reset(train_mode=train_mode)[brain_name]
    n_agents = len(env_info.agents)

    # run training
    train_actor_critic_mind(env, ppo_mind, n_agents,
        n_episodes = arglist.n_episodes,
        train_mode=train_mode)

    # clean up
    env.close()


