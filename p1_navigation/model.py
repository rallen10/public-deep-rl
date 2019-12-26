import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, observation_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            observation_size (int): Dimension of each observation
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # Define hidden layer sizes
        h_layers = [128, 64, 32]
        layer_sizes = zip(h_layers[:-1], h_layers[1:])

        # create sequence of layers
        self.hidden_layers = nn.ModuleList([nn.Linear(observation_size, h_layers[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # creat output
        self.output = nn.Linear(h_layers[-1], action_size)


    def forward(self, observation):
        """Build a network that maps observation -> action values."""
        
        # forward pass through each hidden layer with a ReLU activation function
        x = observation
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)

        # pass through output layer
        # Note: the output should be interpreted the q-value of each action
        x = self.output(x)

        return x
