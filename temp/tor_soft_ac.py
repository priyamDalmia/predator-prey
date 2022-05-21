import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distribution as dist
import numpy as np
from data.agent import BaseAgent
import pdb

class NetworkActor(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, 
            lr, network_dims, **kwargs):
        super().__init__()
        pass

    def forward(self, inputs):
        pass

class NetworkCritic(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, lr
            network_dims, **kwargs):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.lr = lr
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        # Checkpoint Variables 
        self.checkpoint = None
        self.checkpoint_name = None
        # Network Layers
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Layers
            clayers = self.network_dims.clayers
            cl_dims = self.network_dims.cl_dims
            for c in range(clayers):
                self.net.append(
                        nn.Conv2d(in_channels=idim,
                            out_channels=cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = cl_dims[c]
            self.net.append(nn.Flatten())
            #### Must be modified if the network parameters change
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            idim = nl_dims[l]
        self.critic_layer = nn.Linear(nl_dims[-1], 1)
        self.net.append(self.critic_layer)
        self.optimizer = optim.Adam(self.parameters(),
                lr=self.lr, betas=(0.9, 0.99), eps=1e-3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        return x

    def save_state(self, cname):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

class ValueNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, network_dims,
            lr, **kwargs):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        # Checkpoint Variables 
        self.checkpoint = None
        self.checkpoint_name = None
        # Network Layers
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Layers
            clayers = self.network_dims.clayers
            cl_dims = self.network_dims.cl_dims
            for c in range(clayers):
                self.net.append(
                        nn.Conv2d(in_channels=idim,
                            out_channels=cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = cl_dims[c]
            self.net.append(nn.Flatten())
            #### Must be modified if the network parameters change
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            idim = nl_dims[l]
        self.critic_layer = nn.Linear(nl_dims[-1], 1)
        self.net.append(self.critic_layer)
        # Must be chabge to incorporate a different learning rate for the values network.
        self.optimizer = optim.Adam(self.parameters(),
                lr=self.lr, betas=(0.9, 0.99), eps=1e-3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        return x

    def save_state(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, network_dims,
            lr, max_action, **kwargs):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.max_action = max_action
        self.network_dims = network_dims 
        


def SACAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            action_space, memory=None, lr=0.01, gamma=0.95, 
            load_model=False, agent_network={}, **kwargs):
        super(ACAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims 
        self.action_space = action_space
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma
        self.memory = memory

        # Initialize the Network.
        network_dims = agent_network.network_dims
        self.network = NetworkActorCritic(input_dims, 
                output_dims, action_space, lr, network_dims)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        selfnetwork = self.network.to(self.device)

    def get_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
