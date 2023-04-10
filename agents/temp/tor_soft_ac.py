import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distribution as dist
import numpy as np
from data.agent import BaseAgent
import pdb

class NetworkActorCritc(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, 
            lr, network_dims, **kwargs):
        super().__init__()
        pass

    def forward(self, inputs):
        pass

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
