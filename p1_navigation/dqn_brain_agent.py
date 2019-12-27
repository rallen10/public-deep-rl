import numpy as np
import random
from collections import namedtuple, deque
import pickle

from model import DeepQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
SAVE_EVERY = 100        # how often to save progress

# Use GPU if it available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BrainAgent():
    """Interacts with and learns from the environment using ML-Agents Brain Object."""

    def __init__(self, name, brain, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            name (str): name of brain
            brain (BrainParameters): Unity ML-Agent brain
            seed (int): random seed
        """
        self.name = name
        self.brain = brain
        self.seed = random.seed(seed)

        # Q-Network
        self.dqn_local = DeepQNetwork(self.brain.vector_observation_space_size, self.brain.vector_action_space_size, seed).to(device)
        self.dqn_target = DeepQNetwork(self.brain.vector_observation_space_size, self.brain.vector_action_space_size, seed).to(device)
        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(self.brain.vector_action_space_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def process_step(self, observation, action, reward, next_observation, done):
        # Save experience in replay memory
        self.memory.add(observation, action, reward, next_observation, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, observation, eps=0.):
        """Returns actions for given observation as per current policy with e-greedy exploration.
        
        Params
        ======
            observation (array_like): current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(observation)
        self.dqn_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.brain.vector_action_space_size))

    def learn(self, experiences, gamma, ddqn=False):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            ddqn (bool): double q-learning
        """
        observations, actions, rewards, next_observations, dones = experiences

        # compute value of action taken
        # Note: dqn outputs value of all actions, so we need to gather the values of just the actions taken
        q_values = self.dqn_local.forward(observations).gather(1,actions)

        # compute target q values
        if ddqn:
            # use deep double q-learning to compute targets
            # - detach the tensor (i.e. return a copy) from the current graph to not mess with gradient
            # of tensor network
            a_max = self.dqn_local.forward(next_observations).detach().argmax(1)[0].unsqueeze(1)
            q_targets_next = self.dqn_target.forward(next_observations).detach()[a_max]
            q_targets = rewards + gamma*q_targets_next*(1-dones)
        else:
            # use vanilla q-learning to compute targets
            # - detach the tensor (i.e. return a copy) from the current graph to not mess with gradient
            # of tensor network
            # - maximize over all possible actions for each input, and reshape
            q_targets_next = self.dqn_target.forward(next_observations).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + gamma*q_targets_next*(1-dones)

        # compute loss
        loss = F.mse_loss(q_targets, q_values)
        
        # reset gradients
        self.optimizer.zero_grad()

        # perform backward step
        loss.backward()

        # peform optimizer stetp
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.dqn_local, self.dqn_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["observation", "action", "reward", "next_observation", "done"])
        self.seed = random.seed(seed)
    
    def add(self, observation, action, reward, next_observation, done):
        """Add a new experience to memory."""
        e = self.experience(observation, action, reward, next_observation, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        observations = torch.from_numpy(np.vstack([e.observation for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_observations = torch.from_numpy(np.vstack([e.next_observation for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (observations, actions, rewards, next_observations, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train_dqn(env, brain_agent, 
    checkpoint_filename='checkpoint.pth', scores_filename='scores.pkl',
    solved_score=13.0, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """train a deep q-learning agent.
    
    Params
    ======
        env (UnityEnvironment): environment (Unity or Gym) in which agent is trained
        brain_agent (BrainAgent): agent being trained
        solved_score (float): score needed to consider environment solved (avg over 100 episodes)
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_agent.name]
        observation = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):

            # brain selects an action based on current observation
            action = brain_agent.act(observation, eps)

            # the action is applied to the environment
            env_info = env.step(action)[brain_agent.name]
            next_observation = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # the agent processes and learns from the information the environment generates in the step
            brain_agent.process_step(observation, action, reward, next_observation, done)

            # the observation is incremented
            observation = next_observation
            score += reward
            if done:
                break 

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % SAVE_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(brain_agent.dqn_local.state_dict(), checkpoint_filename)
            pickle.dump(scores, open(scores_filename, 'wb'))
        if np.mean(scores_window)>=solved_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(brain_agent.dqn_local.state_dict(), checkpoint_filename)
            pickle.dump(scores, open(scores_filename, 'wb'))
            break

    return scores