import argparse
from unityagents import UnityEnvironment
import numpy as np
from actor_critic_mind import ActorCriticMind
from trainer import train_actor_critic_mind

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def parse_args(args_feed):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--exp-name", type=str, default="default", help="name of training experiment")
    parser.add_argument("--policy-lr", type=float, default=2e-4, help="learning rate for policy network")
    parser.add_argument("--value-lr", type=float, default=1e-3, help="learning rate for value network")
    parser.add_argument("--gae-lambda", type=float, default=0.9, help="lambda value in generalized advantage estimate")
    parser.add_argument("--gae-gamma", type=float, default=1.0, help="gamma value in generalized advantage estimate")
    parser.add_argument("--batch-size", type=int, default=512, help="batch size for training")
    parser.add_argument("--minibatch-size", type=int, default=32, help="minibatch size for training")
    parser.add_argument("--n-epochs", type=int, default=2, help="training epochs per batch")
    parser.add_argument("--n-episodes", type=int, default=1000000, help="training epochs per batch")
    return parser.parse_args(args_feed)

if __name__ == "__main__":
    arglist = parse_args(sys.argv[1:])

    # create environment
    env = UnityEnvironment(file_name='./Tennis_Linux/Tennis.x86_64', no_graphics=True)

    # create a learning agent ('mind')
    brain_name = env.brain_names[0]
    ppo_mind = ActorCriticMind(
        brain_name, 
        env.brains[brain_name], 
        policy_lr=arglist.policy_lr, 
        value_lr=arglist.value_lr)

    # get number of agents
    train_mode = True
    env_info = env.reset(train_mode=train_mode)[brain_name]
    n_agents = len(env_info.agents)

    # run training
    train_actor_critic_mind(env, ppo_mind, n_agents,
        n_episodes = arglist.n_episodes,
        save_rate = 1,
        n_epochs = arglist.n_epochs,
        desired_batch_size = arglist.batch_size,
        desired_minibatch_size = arglist.minibatch_size,
        gamma = arglist.gae_gamma,
        lam = arglist.gae_lambda,
        prefix = arglist.exp_name,
        beta_start = 0.1, beta_end = 0.0001, beta_decay = 0.99,
        epsilon_start=0.3, epsilon_end=0.1, epsilon_decay = 0.995,
        v_epsilon_start=0.3, v_epsilon_end=0.1, v_epsilon_decay=0.995,
        train_mode=train_mode)

    # clean up
    env.close()


