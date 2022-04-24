import torch
import torch.nn as nn
import torch.optim as optim
import torch.distribution as dist
import numpy as np
from data.agent import BaseAgent
import pdb

class NetworkCritic(nn.Module):
    def __init(self, ):
        super(NetworkCritc, self).__init__()
        

        ## Sees what all agents See!
        ## Simply Pass Combined Obseravtion
        pass

    def forward(self, inputs):
        pass

    def train_step(self):
        pass

    def save_checkpoint(self):
        pass

    def save_model(self):
        pass


class NetworkActor(nn.Module):
    def __init__(self, ):
        super(NetworkActor, self).__init__()
        pass

    def forward(self, inputs):
        pass
    

class CACAgent():
    def __init__(self, _id, input_dims, output_dims, 
            action_space, memory=None, lr=0.01, gamma=0.95,
            load_model=False, agent_network={}, **kwargs):
        super(CACAgent, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma 
        self.memory = memory
        self.total_loss = 0
        self.checkpoint = None
        self.checkpoint_name = None
        # Initialize the CAC Network 
        self.actor_network = None
        self.critic_network = None
        if self.load_model:
            # Load Models for both Actor and Critic Here!
            pass
        else:
            self.network_dims = agent_network.network_dims
            self.actor_network = NetworkActor(input_dims, output_dims, action_space,
                    lr, self.network_dims)
            if "critic_network" not in self.agent_network:
                raise "Critic Network is not provided!"
            self.critic_network = self.agent_network["critic_network"]

        self.device = torch.device()
        self.network = self.network.to(self.device)
    
    def get_action(self, observation):
        observation = torch.as_tensor()
        pass

    def train_step(self):
        pass
