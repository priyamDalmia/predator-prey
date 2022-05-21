import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from data.helpers import dodict
from data.agent import BaseAgent

import pdb

class Network(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, 
            lr, network_dims, **kwargs):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space 
        self.lr = lr
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        kwargs = dodict(kwargs)
        # Network Layers
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Network
            clayers = network_dims.clayers
            cl_dims = network_dims.cl_dims
            for c in range(clayers):
                self.net.append(
                        nn.Conv(in_channels=cl_dims[c],
                            out_channels=cl_dims[c+1],
                            kernel_size=1,
                            stride=1))
                self.net.append(nn.Flatten())
            idim = (clayers[-1] * (self.input_dims[-1]**2))
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            idim = nl_dims[l]
        self.net.append(nn.Linear(nl_dims[-1], self.output_dims))
        # Model config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        return F.softmax(x, dim=1) 

class RFAgent(BaseAgent):
    def  __init__(self, _id, input_dims, output_dims,
            action_space, memory=None, lr=0.001, gamma=0.95, 
            load_model=False, agent_network={}, **kwargs):
        """
        An vanilla REINFORCE agent with a policy gradient network.
        Args:
            _id: game agent _id
            input_dims: tuple
            output_dims: int
            action_space: list 
            load_model: bool
            memory: ReplayBuffer 
            lr: float 
            gamma: float
            epsilon: float
            epsilon_end: float 
            epsilon_dec: float

        """
        super(RFAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        # Init Networks

        if self.load_model:
            pass
        network_dims = agent_network.network_dims
        self.network = Network(input_dims, output_dims, action_space, 
                self.lr, network_dims)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.network = self.network.to(self.device)            
    
    def get_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs = self.network(observation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def train_on_batch(self):
        self.network.optimizer.zero_grad()
        rewards, action_probs = self.memory.sample_transition()     
        new_rewards = []
        _sum = 0
        for i in range(len(rewards)):
            r = rewards[-1+i]
            _sum *= self.gamma
            _sum += r
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        # Calculate Discounted rewards
        dis_rewards = torch.as_tensor(new_rewards, dtype=torch.float32, device=self.device)
        loss_actions = dis_rewards.unsqueeze(1) * torch.stack(action_probs)
        loss = (-1) * loss_actions.sum()
        loss.backward()
        self.network.optimizer.step()
        return loss.item()
    
    def save_state(self):
        pass

    def store_transition(self, state, action, reward, _next, done, probs):
        self.memory.store_transition(state, action, reward, _next, done, probs=probs)
    
    def update_eps(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
