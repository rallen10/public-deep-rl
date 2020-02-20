import numpy as np
import random
import pickle

import torch
import torch.optim as optim 
# hack for relative imports in interactive notebook
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils.utils as U
from utils.network_models import DeepNetwork, GaussianActorNetwork

# Use GPU if it available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_trajectories(env, agent_mind, n_agents, desired_batch_size, train_mode):
    """ create observation, actions, rewards, and action probability trajectories from parallel environments

    Params
    ======
        env (): parallel environments
        agent_mind (ActorCriticMind): mind used to control (parallel) agents in environment
        n_agents (int): number of parallel instances
        desired_batch_size (int): maximum timesteps per batch

    Returns
    =======
        observe_list (List[array]): index t holds the n_agents parallel observations at timestep t
        act_log_prob_list (List[Tensor]): index t holds the n_agents parallel log action probabilities at timestep t
        action_list (List[Tensor]): index t holds the n_agents parallel actions at timestep t
        reward_list (List[array]): index t holds the n_agents parallel reward at timestep t
        value_list (List[Tensor]): index t holds the n_agents parallel value estimate at timestep t
    """


    # initialize returning lists
    observe_list=[]
    act_log_prob_list=[]
    action_list=[]
    reward_list=[]
    value_list=[]
    done_list=[]
    batch_episode_count = 1
    batch_episode_scores = []
    episode_rewards = []


    # reset environments and get initial observations
    env_info = env.reset(train_mode=train_mode)[agent_mind.name]
    observations = env_info.vector_observations

    # step through episode
    for t in range(desired_batch_size):

        # select actions based on observations
        actions, action_log_probs = agent_mind.select_action(observations)
        assert actions.shape == (n_agents, agent_mind.brain.vector_action_space_size)
        assert action_log_probs.shape == (n_agents,)

        # apply actions to step environment, store observations
        env_info = env.step(np.clip(actions.cpu().numpy(), -1, 1))[agent_mind.name]
        next_observations = env_info.vector_observations
        assert next_observations.shape == (n_agents, agent_mind.stacked_observation_size)

        # Combine rewards and dones since this is a joint reward
        assert len(env_info.rewards) ==  n_agents
        rewards = np.sum(env_info.rewards)*np.ones(n_agents)

        # record trajectory step
        observe_list.append(observations)
        act_log_prob_list.append(action_log_probs)
        action_list.append(actions)
        reward_list.append(rewards)
        done_list.append(env_info.local_done)
        episode_rewards.append(rewards)
        # scores += env_info.rewards

        # estimate the value of the current observations 
        # with the current value network
        value_list.append(agent_mind.estimate_value(observations))

        # increment observations
        observations = next_observations

        # Handle end of episodes
        if np.any(env_info.local_done):
        
            batch_episode_count += 1
            episode_rewards = np.asarray(episode_rewards)
            assert episode_rewards.shape[1] == n_agents 
            batch_episode_scores.append(np.max(np.sum(episode_rewards, axis=0)))
            episode_rewards = []

            # reset env
            env_info = env.reset(train_mode=train_mode)[agent_mind.name]
            observations = env_info.vector_observations


    next_dones = env_info.local_done
    return {"observe_list":observe_list, 
            "act_log_prob_tensors": act_log_prob_list, 
            "action_tensors": action_list, 
            "reward_list": reward_list, 
            "value_tensors": value_list,
            "done_list": done_list,
            "next_observations": next_observations,
            "next_dones": next_dones,
            "batch_episode_count": batch_episode_count,
            "batch_episode_scores": batch_episode_scores}

def clipped_policy_surrogate(policy, observations, act_log_probs, 
    actions, advantages, epsilon):
    ''' clipped surrogate loss function from batch of episodes

    Params:
    =======
    policy (DeepNetwork): policy network
    observations (Tensor): observation minibatch
        element at [mbi][i] is the observation of agent i 
        from minibatch index mbi
    act_log_probs (Tensor): action log probability minibatch
    actions (Tensor): actions minibatch
    advantages (Tensor): advantages minibatch
    epsilon (float): clippiing factor

    Returns:
    ========


    Notes:
    ======
    Reference: https://arxiv.org/pdf/1707.06347.pdf
    '''

    # get shape to check for consistency
    batch_shape = act_log_probs.shape
    assert advantages.shape == batch_shape
    assert observations.shape[0:2] == batch_shape[0:2]
    assert actions.shape[0:2] == batch_shape[0:2]

    # compute action log probabilities and entropy under new policy
    _, new_act_log_probs, entropy = policy.forward(observations, action=actions)
    assert new_act_log_probs.shape == batch_shape
    assert entropy.shape == actions.shape
    
    # compute policy ratio
    ratio = torch.exp(new_act_log_probs - act_log_probs)
    assert ratio.shape == batch_shape

    # clipped surrogate loss value
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_L = torch.min(ratio*advantages, clip*advantages)
    assert clip.shape == clipped_L.shape == batch_shape
    
    return clipped_L.mean(), entropy.mean()
    

def train_actor_critic_mind(env, agent_mind, n_agents, gamma=0.995, lam=0.9,
    prefix='default', solved_score=30.0, solved_window=100, stop_solved=False, train_mode=True,
    n_episodes=2000, n_epochs=4, desired_batch_size=64, desired_minibatch_size=16, save_rate = 1,
    beta_start=0.01, beta_end=0.0001, beta_decay=0.995,
    epsilon_start=0.3, epsilon_end=0.01, epsilon_decay=0.995,
    v_epsilon_start=0.3, v_epsilon_end=0.01, v_epsilon_decay=0.995,
    max_grad_norm=0.5):
    """train a PPO actor-critic agent in parallel environments
    
    Params
    ======
        env (UnityEnvironment): environment (Unity or Gym) in which agent is trained
        agent_mind (ActorCriticMind): mind used to control (parallel) agents in environment
        n_agents (int): number of agents controlled by the agent_mind
        gamma (float): return discounting factor
        lam (float): general advantage function parameter
        prefix (str): name used to create save files
        solved_score (float): score needed to consider environment solved 
        solved_window (int): number of episodes to average over to determine if solved
        stop_solved (bool): stop training when solved
        train_mode (bool); whether training should occur (otherwise just used to visualize policy)
        n_episodes (int): maximum number of training episodes
        desired_batch_size (int):
        desired_minibatch_size (int): 
        save_rate
        n_epochs (int): number of epochs to train per batch
        beta_start (float): starting value of beta, for entropy exploration
        beta_end (float): minimum value of beta
        beta_decay (float): multiplicative factor (per episode) for decreasing beta
        epsilon_start (float): starting value of epsilon, for clipping fraction
        epsilon_end (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per batch) for decreasing epsilon
        v_epsilon_start (float): starting value of value clipping fraction
        v_epsilon_end (float): minimum value of value clipping fraction
        v_epsilon_decay (float): multiplicative factor (per batch) for decreasing v_epsilon
    """

    def save_training():
            torch.save(agent_mind.policy_network.state_dict(), policy_checkpoint_filename)
            torch.save(agent_mind.value_network.state_dict(), value_checkpoint_filename)
            # TODO: save epsilon and beta
            pickle.dump({   "scores":score_history, 
                            "policy_loss":policy_loss_history, 
                            "value_loss":value_loss_history,
                            "clipped_surrogate":clipped_L_history,
                            "entropy":entropy_history,
                            "clip_factor":epsilon_history,
                            "value_clip_factor":v_epsilon_history,
                            "entropy_coef":beta_history,
                            "episode_counts": episode_history}, open(scores_filename, 'wb'))

    # check inputs
    if desired_minibatch_size > desired_batch_size: raise ValueError

    # Establish storage to track progress over episodes
    score_history = []
    policy_loss_history = []
    value_loss_history = []
    clipped_L_history = []
    entropy_history = []
    epsilon_history = []
    v_epsilon_history = []
    beta_history = []
    episode_count = 0
    batch_count = 0
    episode_history = []

    solved = False
    policy_checkpoint_filename = prefix + '_policy_checkpoint.pth'
    value_checkpoint_filename = prefix + '_value_checkpoint.pth'
    scores_filename = prefix + '_scores.pkl'


    # set up learning parameters
    beta = beta_start
    epsilon = epsilon_start
    v_epsilon = v_epsilon_start

    # loop through training batches (may be more than one per episode)
    while True:
        batch_count += 1

        # collect batch of trajectories trajectories
        batch_data = collect_trajectories(
            env=env,  
            agent_mind=agent_mind, 
            n_agents=n_agents, 
            desired_batch_size=desired_batch_size,
            train_mode=train_mode)
        observe_list = batch_data['observe_list']
        act_log_prob_list = batch_data["act_log_prob_tensors"] 
        action_list = batch_data["action_tensors"] 
        reward_list = batch_data["reward_list"] 
        value_list = batch_data["value_tensors"]
        done_list = batch_data["done_list"]
        next_observations = batch_data["next_observations"]
        next_dones = batch_data["next_dones"]
        episode_count += batch_data["batch_episode_count"]
        episode_history.append(episode_count)
        score_history.extend(batch_data["batch_episode_scores"])


        # inspect batch data to make sure outputs are of appropriate length
        batch_size = len(observe_list)
        assert batch_size == desired_batch_size
        batch_indices = np.arange(batch_size)
        assert (batch_size == len(act_log_prob_list) 
                        == len(action_list) 
                        == len(reward_list) 
                        == len(value_list) 
                        == len(done_list))


        if episode_count >= n_episodes:
            break


        if train_mode:
            # convert lists of tensors to arrays for advantage calc
            reward_arrays = np.asarray(reward_list).transpose()
            done_arrays = np.asarray(done_list).transpose()
            value_arrays = torch.stack(value_list).cpu().numpy().transpose() # need to copy tensor to cpu before converting to numpy
            next_values = agent_mind.estimate_value(next_observations).cpu().numpy()
            assert reward_arrays.shape == value_arrays.shape == done_arrays.shape == (n_agents, batch_size)
            assert len(next_values) == len(next_dones) == n_agents

            # compute advantages and returns at each time step
            advantage_list = []
            returns_list = []
            for rew, val, done, next_val, next_done in zip(reward_arrays, value_arrays, done_arrays, next_values, next_dones):
                advs, rets, _ = U.general_advantage_estimation(
                    rewards=rew,
                    values=val,
                    dones=done,
                    lam=lam,
                    gamma=gamma,
                    next_value=next_val,
                    next_done=next_done)
                advantage_list.append(advs)
                returns_list.append(rets)
            # returns_arrays = np.array([U.episode_returns(rew,gamma) for rew in reward_arrays]).transpose()
            advantage_arrays = np.asarray(advantage_list).transpose()
            returns_arrays = np.asarray(returns_list).transpose()
            assert advantage_arrays.shape == (batch_size, n_agents)  # just checking my own logic
            assert returns_arrays.shape == (batch_size, n_agents)

            # format training batch data
            observe_batch = torch.from_numpy(np.asarray(observe_list)).float().to(DEVICE)
            action_batch = torch.stack(action_list).to(DEVICE)
            advantage_batch = torch.from_numpy(advantage_arrays).unsqueeze(-1).float().to(DEVICE)
            value_batch = torch.stack(value_list).unsqueeze(-1).float().to(DEVICE)

            # detach so target variables don't effect gradient (?? not sure I fully understand this, or if using correctly ??)
            returns_batch = torch.from_numpy(returns_arrays).float().unsqueeze(-1).to(DEVICE).detach()
            act_log_prob_batch = torch.stack(act_log_prob_list).unsqueeze(-1).float().to(DEVICE).detach()

            assert observe_batch.shape == (batch_size, n_agents, agent_mind.stacked_observation_size)
            assert action_batch.shape == (batch_size, n_agents, agent_mind.brain.vector_action_space_size)
            assert act_log_prob_batch.shape == (batch_size, n_agents, 1)
            assert advantage_batch.shape == (batch_size, n_agents, 1)
            assert returns_batch.shape == (batch_size, n_agents, 1)
            assert value_batch.shape == (batch_size, n_agents, 1)

            # iterate through epochs of training batch to update policy and value networks
            for epoch in range(n_epochs):

                # randomize batch ordering
                np.random.shuffle(batch_indices)


                # iterate through minibatches
                for mbi_start in range(0, batch_size, desired_minibatch_size):

                    # get actual minibatch size
                    minibatch_size = desired_minibatch_size if (mbi_start + desired_minibatch_size <= batch_size) else batch_size - mbi_start
                    checkvar1 = np.random.randint(minibatch_size)
                    checkvar2 = np.random.randint(n_agents)

                    # format training batch into minibatches
                    observe_minibatch = observe_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    action_minibatch = action_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    act_log_prob_minibatch = act_log_prob_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    advantage_minibatch = advantage_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    returns_minibatch = returns_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    value_minibatch = value_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]

                    # perform checks on minibatch formation
                    # NOTE: this would be rmoved for production code, 
                    # but it's great for testing logic and debugging in
                    # prototype code
                    assert observe_minibatch.shape == (minibatch_size, n_agents, agent_mind.stacked_observation_size)
                    assert action_minibatch.shape == (minibatch_size, n_agents, agent_mind.brain.vector_action_space_size)
                    assert act_log_prob_minibatch.shape == (minibatch_size, n_agents, 1)
                    assert advantage_minibatch.shape == (minibatch_size, n_agents, 1)
                    assert returns_minibatch.shape == (minibatch_size, n_agents, 1)
                    assert value_minibatch.shape == (minibatch_size, n_agents, 1)
                    assert all(np.isclose(observe_minibatch[0][0], observe_batch[batch_indices[mbi_start]][0]))
                    assert all(np.isclose(observe_minibatch[checkvar1][checkvar2], observe_batch[batch_indices[mbi_start + checkvar1]][checkvar2]))
                    assert all(np.isclose(action_minibatch[checkvar1][checkvar2], action_batch[batch_indices[mbi_start + checkvar1]][checkvar2]))
                    assert np.isclose(act_log_prob_minibatch[checkvar1][checkvar2], act_log_prob_batch[batch_indices[mbi_start + checkvar1]][checkvar2])
                    assert np.isclose(advantage_minibatch[checkvar1][checkvar2], advantage_batch[batch_indices[mbi_start + checkvar1]][checkvar2])
                    assert np.isclose(returns_minibatch[checkvar1][checkvar2], returns_batch[batch_indices[mbi_start + checkvar1]][checkvar2])
                    assert np.isclose(value_minibatch[checkvar1][checkvar2], value_batch[batch_indices[mbi_start + checkvar1]][checkvar2])

            
                    # compute surrogate policy loss function
                    clipped_L, entropy = clipped_policy_surrogate(
                            policy = agent_mind.policy_network, 
                            observations = observe_minibatch, 
                            act_log_probs = act_log_prob_minibatch, 
                            actions = action_minibatch, 
                            advantages = advantage_minibatch,
                            epsilon=epsilon)
                    policy_loss = -clipped_L - beta*entropy

                    # Policy update step
                    agent_mind.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent_mind.policy_network.parameters(),
                                             max_grad_norm)
                    agent_mind.policy_optimizer.step()

                    # Record policy loss and clean up
                    clipped_L_history.append(torch.mean(clipped_L).clone().cpu().detach().numpy())
                    entropy_history.append(torch.mean(entropy).clone().cpu().detach().numpy())
                    policy_loss_history.append(policy_loss.data.clone().cpu().detach().numpy()) # send to cpu so it can be opened on non-gpu device
                    del policy_loss

                    # compute value loss using mean squared error compared with emperical returns
                    vpred = agent_mind.value_network.forward(observe_minibatch)
                    vpred_clipped = value_minibatch + torch.clamp(vpred - value_minibatch, -v_epsilon, v_epsilon)
                    vloss1 = torch.pow(vpred - returns_minibatch, 2)
                    vloss2 = torch.pow(vpred_clipped - returns_minibatch, 2)
                    value_loss = torch.mean(torch.max(vloss1, vloss2))

                    assert vpred.shape == (minibatch_size, n_agents, 1)
                    assert vpred_clipped.shape == (minibatch_size, n_agents, 1)
                    assert vloss1.shape == (minibatch_size, n_agents, 1)
                    assert vloss2.shape == (minibatch_size, n_agents, 1)


                    # Value network update step
                    agent_mind.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent_mind.value_network.parameters(),
                                             max_grad_norm)
                    agent_mind.value_optimizer.step()
                    value_loss_history.append(value_loss.data.clone().cpu().detach().numpy()) # send to cpu so it can be opened on non-gpu device
                    del value_loss

            # update learning parameters: epsilon, beta
            epsilon_history.append(epsilon)
            beta_history.append(beta)
            v_epsilon_history.append(v_epsilon)
            epsilon = max(epsilon*epsilon_decay, epsilon_end)
            v_epsilon = max(v_epsilon*v_epsilon_decay, v_epsilon_end)
            beta = max(beta*beta_decay, beta_end)

            # save and display progress
            if episode_count % save_rate == 0:
                print("Episode: {:d}, training batch: {:d}, score: {:f}, policy loss: {:f}, value loss: {:f}".format(
                    episode_count, batch_count, 
                    np.mean(score_history[max(0,len(score_history)-solved_window):]), 
                    np.mean(policy_loss_history[max(0,len(policy_loss_history)-solved_window):]), 
                    np.mean(value_loss_history[max(0,len(value_loss_history)-solved_window):])))
                save_training()

            # check for termination condition
            if (len(score_history) >= solved_window and 
                np.mean(score_history[-solved_window:]) > solved_score and
                not solved):

                solved = True
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    episode_count-solved_window, 
                    np.mean(score_history[-solved_window:])))
                save_training()

                if stop_solved:
                    break
                else:
                    print('\nContinuing training...')
