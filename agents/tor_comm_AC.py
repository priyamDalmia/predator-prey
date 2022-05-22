import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from data.helplers import dodict
from data.agents import BaseAgent
import pdb

class NetworkCommAC(nn.Module):
    def __init__(self,
            input_dims,
            output_dims,
            action_space,
            network_dims, 
            num_agents,
            **kwargs):
        super(NetworkCommAC, self).__init__()
        self.input_dims = input_dims
        self.comm_dims = comm_dims
        self.action_space = action_space
        self.num_agents = num_agents
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        self.nlayers = network_dims.nlayers
        self.nl_dims = network_dims.nl_dims
        self.clayers = network_dims.clayers
        self.cl_dims = network_dims.cl_dims
        self.rnn_layers = network_dims.rnn_layers
        # NETWORK ARCHITECTURE
        idim = self.input_dims[0]
        for c in range(self.clayers):
            self.net.append(
                    nn.Conv2d(
                        in_channels=idim,
                        out_channels=self.cl_dims[c],
                        kernel_size=(2,2),
                        stride=1))
            idim = self.cl_dims[c]
        self.net.append(nn.Flatten())
        idim = self.cl_dims[-1] * \
                ((self.input_dims[-1]-int(self.clayers))**2)
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim, self.ml_dims[l]))
            idim = self.nl_dims[l]
        self.comm_layer = nn.Linear(self.nl_dims[-1], self.ouptut_dims)

    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        comm = self.

class NetworkActorCritic(nn.Module):
    def __init__(self,
            input_dims,
            output_dims,
            action_space,
            network_dims, 
            num_agents,
            **kwargs):
        super(NetworkCommAC, self).__init__()
        self.input_dims = input_dims
        self.comm_dims = comm_dims
        self.action_space = action_space
        self.num_agents = num_agents
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        self.nlayers = network_dims.nlayers
        self.nl_dims = network_dims.nl_dims
        self.clayers = network_dims.clayers
        self.cl_dims = network_dims.cl_dims
        self.rnn_layers = network_dims.rnn_layers
        # NETWORK ARCHITECTURE
        idim = self.input_dims[0]
        for c in range(self.clayers):
            self.net.append(
                    nn.Conv2d(
                        in_channels=idim,
                        out_channels=self.cl_dims[c],
                        kernel_size=(2,2),
                        stride=1))
            idim = self.cl_dims[c]
        self.net.append(nn.Flatten())
        idim = self.cl_dims[-1] * \
                ((self.input_dims[-1]-int(self.clayers))**2)
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim, self.ml_dims[l]))
            idim = self.nl_dims[l]
        self.comm_layer = nn.Linear(self.nl_dims[-1], self.ouptut_dims)

    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x


class cAACAgent(BaseAgent):
    def __init__(self,
            _id,
            input_dims,
            output_dims,
            action_space,
            memory=False,
            lr=0.005,
            gamma=0.95,
            load_model=False,
            eval_model=False,
            agent_network={},
            **kwargs):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.load_model = load_model
        self.initialize_memory(self.input_dims)
        self.deivce = torch.device('cuda:0'\
                if torch.cuda.is_available() else 'cpu')
        if load_model:
            pass
    
    def get_action(self, observation, _id, message_last):
        # Combine Messages and Observation here.
        observation = torch.as_tensor(observation, dtype=torch.float32)
        message = torch.as_tensor(observation, dtype=torch.float32)
        probs, values, hidden_state, message = self.network(observation.unsqueeze(0), message, _id=_id)
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        return action.item(), hidden_state, message
    
    def store_transition():
        pass
