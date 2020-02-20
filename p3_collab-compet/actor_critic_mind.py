import random
import torch
import torch.optim as optim 

# hack for relative imports in interactive notebook
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
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
        self.stacked_observation_size = (self.brain.vector_observation_space_size * 
                                        self.brain.num_stacked_vector_observations)
        self.policy_network = GaussianActorNetwork(self.stacked_observation_size, self.brain.vector_action_space_size, policy_h_layers, seed).to(DEVICE)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # Value network maps observations to state value approximations
        self.value_network = DeepNetwork(self.stacked_observation_size, 1, value_h_layers, seed).to(DEVICE)
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