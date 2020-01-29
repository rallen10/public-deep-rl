import numpy as np
import random
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim 

from warnings import warn

# hack for relative imports in interactive notebook
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils.utils as U
from utils.network_models import DeepNetwork, GaussianActorNetwork

# Use GPU if it available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCriticMind():
    """Interacts with and learns from the environment using ML-Agents Brain Object"""

    def __init__(self, name, brain, 
        policy_h_layers=[256, 64, 16], value_h_layers=[256, 64, 16],
        policy_lr=5e-4, value_lr=1e-3,
        seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            name (str): name of brain
            brain (BrainParameters): Unity ML-Agent brain
            policy_h_layers (List[int]): size of policy network hidden layers
            value_h_layers (List[int]): size of value network hidden layers
            seed (int): random seed
            policy_lr (float): policy learning rate
            value_lr (float): value learning rate
        """

        self.name = name
        self.brain = brain
        self.seed = random.seed(seed)

        # Policy network maps observations to actions
        self.policy_network = GaussianActorNetwork(self.brain.vector_observation_space_size, self.brain.vector_action_space_size, policy_h_layers, seed).to(DEVICE)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # Value network maps observations to state value approximations
        self.value_network = DeepNetwork(self.brain.vector_observation_space_size, 1, value_h_layers, seed).to(DEVICE)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_lr)
        # self.value_criterion = torch.nn.MSELoss()

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def select_action(self, observations):
        """ generate actions for parallel agents based on observations

        Params
        ======
            observations (List[np.array]): list of arrays containing observations for each agent

        Returns
        =======
            action_values (Tensor): array of actions, one for each agent
            action_log_probs (Tensor): array of action log probabilities
        """

        observations = torch.from_numpy(observations).float().unsqueeze(0).to(DEVICE)
        self.policy_network.eval()  # switch to eval mode
        with torch.no_grad():       # this might be redundant with .eval()
            # print("action selection observation shape: {}".format(observations.shape))
            actions, action_log_probs, _ = self.policy_network.forward(observations)
        self.policy_network.train() # set back into training mode
        assert actions.shape[0:2] == action_log_probs.shape[0:2] == observations.shape[0:2]

        return actions.squeeze(), action_log_probs.squeeze()

    def estimate_value(self, observations):
        """ estimate value of each parallel agent's observations

        Params
        ======
            observations (List[np.array]): list of arrays containing observations for each agent

        Returns
        =======
            values (Tensor): array of value estimates, on for each agent
        """
        observations = torch.from_numpy(observations).float().unsqueeze(0).to(DEVICE)

        self.value_network.eval()  # switch to eval mode
        with torch.no_grad():       # this might be redundant with .eval()
            values =  self.value_network(observations)
        self.value_network.train() # set back into training mode

        return values.squeeze()

def collect_trajectories(env, agent_mind, n_agents, desired_batch_size, batch_init_observations, reset_environment):
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
    episode_done = False
    # scores = np.zeros(n_agents)

    # reset environments and get initial observations
    observations = batch_init_observations
    if reset_environment: 
        env_info = env.reset(train_mode=True)[agent_mind.name]
        observations = env_info.vector_observations

    # step through episode
    for t in range(desired_batch_size):

        # select actions based on observations
        actions, action_log_probs = agent_mind.select_action(observations)

        # apply actions to step environment, store observations
        env_info = env.step(np.clip(actions.cpu().numpy(), -1, 1))[agent_mind.name]
        next_observations = env_info.vector_observations

        # record trajectory step
        observe_list.append(observations)
        act_log_prob_list.append(action_log_probs)
        action_list.append(actions)
        reward_list.append(env_info.rewards)
        done_list.append(env_info.local_done)
        # scores += env_info.rewards

        # estimate the value of the current observations 
        # with the current value network
        value_list.append(agent_mind.estimate_value(observations))

        # increment observations
        observations = next_observations

        # break if any env is done
        if np.any(env_info.local_done):
            episode_done = True
            break

    next_dones = env_info.local_done
    return {"observe_list":observe_list, 
            "act_log_prob_tensors": act_log_prob_list, 
            "action_tensors": action_list, 
            "reward_list": reward_list, 
            "value_tensors": value_list,
            "done_list": done_list,
            "next_observations": next_observations,
            "next_dones": next_dones,
            "episode_done": episode_done}

def clipped_policy_surrogate(policy, observations, act_log_probs, actions, advantages,
                      epsilon, beta):
    ''' clipped surrogate loss function from batch of episodes

    Params:
    =======
    policy (DeepNetwork): policy network
    observations (Tensor): observation batch, 
        index t holds the n_agents parallel observations at timestep t
    act_log_probs (Tensor): action log probability batch
    actions (Tensor): actions batch
    advantages (Tensor): advantages batch
    epsilon (float): clippiing factor
    beta (float): entropy weighting faction (exploration)

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
    # print("surrogate observation shape: {}".format(observations.shape))
    _, new_act_log_probs, entropy = policy.forward(observations, action=actions)
    assert new_act_log_probs.shape == batch_shape
    assert entropy.shape == actions.shape

    # print("DEBUG: observation shape: {}".format(observations.shape))
    # print("DEBUG: actions shape: {}".format(actions.shape))
    # print("DEBUG: new act log probs shape: {}".format(new_act_log_probs.shape))
    # print("DEBUG: entropy shape: {}".format(entropy.shape))
    # entropy = torch.mean(entropy, dim=1)
    # new_act_log_probs.squeeze_(-1)
    
    # compute policy ratio
    # # ratio =  new_probs/old_probs
    # print("DEBUG: act log probs shape: {}".format(act_log_probs.shape))
    # print("DEBUG: new act log probs shape: {}".format(new_act_log_probs.shape))
    ratio = torch.exp(new_act_log_probs - act_log_probs)
    assert ratio.shape == batch_shape
    # print("DEBUG: ratio shape: {}".format(ratio.shape))

    # clipped surrogate loss value
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_L = torch.min(ratio*advantages, clip*advantages)
    assert clip.shape == clipped_L.shape == batch_shape
    
    # return torch.mean(clipped_L + beta*entropy)
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
        lambda (float): general advantage function parameter
        prefix (str): name used to create save files
        solved_score (float): score needed to consider environment solved (avg over 100 episodes)
        stop_solved (bool): stop training when solved
        n_episodes (int): maximum number of training episodes
        max_episode_len (int): maximum number of timesteps per episode
        n_epochs (int): number of epochs to train per batch
        epsilon_start (float): starting value of epsilon, for clipping fraction
        epsilon_end (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
        beta_start (float): starting value of beta, for entropy exploration
        beta_end (float): minimum value of beta
        beta_decay (float): multiplicative factor (per episode) for decreasing beta
    """

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
    episode_rewards = []
    next_observations = None
    episode_done = True

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

        # print("Training Iteration: {}".format(batch_iter))

        # collect batch of trajectories trajectories
        batch_data = collect_trajectories(
            env=env,  
            agent_mind=agent_mind, 
            n_agents=n_agents, 
            desired_batch_size=desired_batch_size,
            batch_init_observations=next_observations,
            reset_environment=episode_done)
        observe_list = batch_data['observe_list']
        act_log_prob_list = batch_data["act_log_prob_tensors"] 
        action_list = batch_data["action_tensors"] 
        reward_list = batch_data["reward_list"] 
        value_list = batch_data["value_tensors"]
        done_list = batch_data["done_list"]
        next_observations = batch_data["next_observations"]
        next_dones = batch_data["next_dones"]
        episode_done = batch_data["episode_done"]
        episode_rewards.extend(reward_list)
        episode_history.append(episode_count)


        # inspect batch data to make sure outputs are of appropriate length
        batch_size = len(observe_list)
        assert (batch_size == desired_batch_size and not batch_data['episode_done']) or batch_data['episode_done']
        batch_indices = np.arange(batch_size)
        assert (batch_size == len(act_log_prob_list) 
                        == len(action_list) 
                        == len(reward_list) 
                        == len(value_list) 
                        == len(done_list))

        # Handle end of episodes
        if episode_done: 
            episode_count += 1
            assert len(episode_rewards) == 1001, "episode reward len = {}".format(len(episode_rewards))
            score_history.append(np.mean(np.sum(episode_rewards, axis=0)))
            episode_rewards = []

            # save and display progress
            # print("ep count: {}, save_rate: {}".format(episode_count, save_rate))
            if episode_count % save_rate == 0:
                print("Episode: {:d}, training batch: {:d}, score: {:f}, policy loss: {:f}, value loss: {:f}".format(
                    episode_count, batch_count, score_history[-1], policy_loss_history[-1], value_loss_history[-1]))
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

            assert observe_batch.shape == (batch_size, n_agents, agent_mind.brain.vector_observation_space_size)
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
                    # observe_minibatch = observe_batch[batch_indices[mbi_start:mbi_start+minibatch_size]].reshape(
                    #     minibatch_size*n_agents, agent_mind.brain.vector_observation_space_size)
                    # act_log_prob_minibatch = act_log_prob_batch[batch_indices[mbi_start:mbi_start+minibatch_size]].reshape(
                    #     minibatch_size*n_agents, 1)
                    # action_minibatch = action_batch[batch_indices[mbi_start:mbi_start+minibatch_size]].reshape(
                    #     minibatch_size*n_agents, agent_mind.brain.vector_action_space_size)
                    # advantage_minibatch = advantage_batch[batch_indices[mbi_start:mbi_start+minibatch_size]].reshape(
                    #     minibatch_size*n_agents, 1)
                    # returns_minibatch = returns_batch[batch_indices[mbi_start:mbi_start+minibatch_size]].reshape(
                    #     minibatch_size*n_agents, 1)
                    # value_minibatch = value_batch[batch_indices[mbi_start:mbi_start+minibatch_size]].reshape(
                    #     minibatch_size*n_agents, 1)
                    # assert all(np.isclose(observe_minibatch[0], observe_batch[mbi_start][0]))
                    # assert all(np.isclose(observe_minibatch[0], observe_batch[mbi_start][0]))
                    observe_minibatch = observe_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    assert observe_minibatch.shape == (minibatch_size, n_agents, agent_mind.brain.vector_observation_space_size)
                    action_minibatch = action_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    assert action_minibatch.shape == (minibatch_size, n_agents, agent_mind.brain.vector_action_space_size)
                    act_log_prob_minibatch = act_log_prob_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    assert act_log_prob_minibatch.shape == (minibatch_size, n_agents, 1)
                    advantage_minibatch = advantage_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    assert advantage_minibatch.shape == (minibatch_size, n_agents, 1)
                    returns_minibatch = returns_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    assert returns_minibatch.shape == (minibatch_size, n_agents, 1)
                    value_minibatch = value_batch[batch_indices[mbi_start:mbi_start+minibatch_size]]
                    assert value_minibatch.shape == (minibatch_size, n_agents, 1)

                    # print("mbi_start: {}".format(mbi_start))
                    # print("minibatch_indices: {}".format(batch_indices[mbi_start:mbi_start+minibatch_size]))
                    # print("checkvar1: {}".format(checkvar1))
                    # print("checkvar2: {}".format(checkvar2))
                    # print("obs checvar mbatch {}\n obs checkvars batch  {}".format(observe_minibatch[checkvar1][checkvar2], observe_batch[mbi_start + checkvar1][checkvar2]))
                    assert all(np.isclose(observe_minibatch[0][0], observe_batch[batch_indices[mbi_start]][0]))
                    assert all(np.isclose(observe_minibatch[checkvar1][checkvar2], observe_batch[batch_indices[mbi_start + checkvar1]][checkvar2]))
                    # assert False
                    # print("mbi_start: {}".format(mbi_start))
                    # print("minibatch_indices: {}".format(batch_indices[mbi_start:mbi_start+minibatch_size]))
                    # print("checkvar1: {}".format(checkvar1))
                    # print("checkvar2: {}".format(checkvar2))
                    # print("action_batch: {}".format(action_batch))
                    # print("action_minibatch: {}".format(action_minibatch))
                    # print(action_minibatch[checkvar1][checkvar2], action_batch[mbi_start + checkvar1][checkvar2])
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
                            epsilon=epsilon, 
                            beta=beta)
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
                    assert vpred.shape == (minibatch_size, n_agents, 1)
                    vpred_clipped = value_minibatch + torch.clamp(vpred - value_minibatch, -v_epsilon, v_epsilon)
                    assert vpred_clipped.shape == (minibatch_size, n_agents, 1)
                    vloss1 = torch.pow(vpred - returns_minibatch, 2)
                    assert vloss1.shape == (minibatch_size, n_agents, 1)
                    vloss2 = torch.pow(vpred_clipped - returns_minibatch, 2)
                    assert vloss2.shape == (minibatch_size, n_agents, 1)
                    value_loss = torch.mean(torch.max(vloss1, vloss2))

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
