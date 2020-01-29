import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNetwork(nn.Module):
    """General network model to be used for policies (actors) and value functions (baselines and critics)."""

    def __init__(self, input_size, output_size, hidden_sizes, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each input (e.g. observations for a policy net)
            output_size (int): Dimension of each ouput (e.g. action for a policy net)
            hidden_sizes (List[int]): hidden layer sizes
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # Stitch together hidden layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])

        # create sequence of layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # creat output
        self.output = nn.Linear(hidden_sizes[-1], output_size)


    def forward(self, inputs):
        """Build a network that maps inputs-> output values."""
        
        # forward pass through each hidden layer with a ReLU activation function
        x = inputs
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)

        # pass through output layer
        # Note: the output should be interpreted the q-value of each action
        x = self.output(x)

        return x

class GaussianActorNetwork(nn.Module):
    """ Actor network that outputs gaussian distributions, good for stochastic continuous actions"""

    def __init__(self, observation_dim, action_dim, hidden_layers, seed):
        """Init params and construct layers of model

        Params
        ======
            observation_dim (init): Dimension of observation space input
            action_dim (int): Dimension of action space output
            hidden_layers (List[int]): hidden layer sizes
            seed (int): random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Stitch together hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        # create sequence of layers
        self.hidden_layers = nn.ModuleList([nn.Linear(observation_dim, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # create output to compute mean and standard deviation of gaussian distribution
        # Note: std of type nn.Parameter is special tensor that is automatically treated as 
        # a model parameter for nn.Module objects
        self.output = nn.Linear(hidden_layers[-1], action_dim)
        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observation, action=None):
        """Build a network that maps inputs-> output values

        Params
        ======
            observation: vector of observations, inputs to network
            action: if action is passed, use forward to compute probability of action
        """
        
        # forward pass through each hidden layer with a ReLU activation function
        x = observation
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)

        # pass through output layer to get mean of distirubtion
        # Note: the output should be interpreted the q-value of each action
        x = self.output(x)

        # compute mean of distribution from output layer
        mean = torch.tanh(x)

        # create output distribution to sample stochastic actions
        # Use the softplus function to map the std parameters to (0, inf)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))

        # if no action is provided, sample from computed distribution
        if action is None:
            action = dist.sample()

        # compute log probability of action
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)

        # compute entropy of action
        entropy = dist.entropy()

        return action, log_prob, entropy 
