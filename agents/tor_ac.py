import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from data.helpers import dodict
from data.agent import BaseAgent

import pdb

class NetworkActor(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, 
            lr, actor_dims, **kwargs):
        super(NetworkActor, self).__init__()
        breakpoint()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space 
        self.lr = lr
        self.actor_dims = actor_dims
        self.net = nn.ModuleList()
        kwargs = dodict(kwargs)
        # Network Layers
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Network
            clayers = kwargs.clayers
            cl_dims = kwargs.cl_dims
            for c in clayers:
                self.net.append(
                        nn.Conv(in_channels=cl_dims[c],
                            out_channels=cl_dims[c+1],
                            kernel_size=1,
                            stride=1))
                self.net.append(F.relu())
                self.net.append(nn.Flatten())
            idim = (clayers[-1] * (self.input_dims[-1]**2))
        nlayers = kwargs.nlayers
        nl_dims = kwargs.nl_dims
        for l in nlayers:
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            self.net.append(F.relu())
            idim = nl_dims[l]
        self.net.append(nn.Linear(nl_dims[-1], self.output_dims))
        # Model config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        return x 

class NetworkCritic(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, 
            critic_dims, lr, **kwargs):
        super(NetworkCritic, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space 
        self.lr = self.lr
        self.critic_dims = critic_dims
        kwargs = dodict(kwargs)
        self.net = nn.ModuleList()
        # Network Layers
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Network
            clayers = kwargs.clayers
            cl_dims = kwargs.cl_dims
            for c in clayers:
                self.net.append(
                        nn.Conv(in_channels=cl_dims[c],
                            out_channels=cl_dims[c+1],
                            kernel_size=1,
                            stride=1))
                self.net.append(F.relu())
                self.net.append(nn.Flatten())
            idim = (clayers[-1] * (self.input_dims[-1]**2))
        nlayers = kwargs.nlayers
        nl_dims = kwargs.nl_dims
        for l in nlayers:
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            self.net.append(F.relu())
            idim = nl_dims[l]
        self.net.append(nn.Linear(nl_dims[-1], self.output_dims))
        # Model config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        return x 

class ActorCriticAgent(BaseAgent):
    def  __init__(self, _id, input_dims, output_dims, 
        """
        An ActorCritc Agent with a policy gradient network.
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
        super(ActorCriticAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        # Init Networks

        if self.load_model:
            pass
        breakpoint()
        actor_net = agent_network.actor_net
        critic_net = agent_network.critic_net
        self.actor = NetworkActor(input_dims, output_dims, action_space, 
                self.lr, actor_net)
        self.critic = NetworkCritic(input_dims, output_dims, action_space, 
                self.lr, critic_net)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.actor = self.actor.to(self.deivce)
        sefl.critic = self.critic.to(self.device)

    @torch.no_grad()
    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        # Get action from actor here.
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        action_probs = self.network(observation.unsqueeze(0))
        action_dist = dist.Categorical(action_probs) 
        breakpoint()
        pass

    def train_on_batch(self):
        if self.memory.counter < 500:
            return 
        breakpoint() 
        pass
    
    def save_state(self):
        pass

    def store_transition(self, state, action, reward, _next, done):
        self.memory.store_transition(state, action, reward, _next, done)
    
    def update_eps(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
