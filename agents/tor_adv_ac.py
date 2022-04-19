import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist 
import numpy as np
from data.agent import BaseAgent
import pdb

class NetworkActorCritic(nn.Module):
    def __init__(self, input_dims, output_dims, action_space,
            lr, network_dims, **kwargs):
        super(NetworkActorCritic, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.lr = lr
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        # Network Layers 
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Layers
            clayers = self.network_dims.clayers
            cl_dims = self.network_dims.cl_dims
            for c in range(clayers):
                self.net.append(
                        nn.Conv2d(in_channels=cl_dims[c],
                            out_channels=cl_dims[c+1],
                            kernel_size=1,
                            stride=1))
            self.net.append(nn.Flatten())
            idim = 300
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            idim = nl_dims[l]
        self.actor_layer = nn.Linear(nl_dims[-1], self.output_dims)
        self.critic_layer = nn.Linear(nl_dims[-1], 1)
        # Model Config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
    
    def forward(self, inputs):
        for layer in self.net:
            x = F.relu(layer(inputs))
            inputs = x
        logits = self.actor_layer(x)
        values = self.critic_layer(x)
        return F.softmax(logits, dim=1), values

class ACAgent(BaseAgent):
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

        # Testing internal memory
        self.actor_loss = []
        self.delta_loss = []
        self.values = []
        self.log_probs = []

        # Initialize the AC network 
        network_dims = agent_network.network_dims
        self.network = NetworkActorCritic(input_dims, output_dims, action_space,
                lr, network_dims)
        self.deivce = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(self.device)

    def get_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs, values = self.network(observation.unsqueeze(0))
        try:
            action_dist = dist.Categorical(probs)
        except Exception as e:
            print(e)
            breakpoint()
        action = action_dist.sample() 
        log_probs =  action_dist.log_prob(action)
        self.values.append(values)
        self.log_probs.append(log_probs)
        return action.item(), log_probs

    def train_on_batch(self):
        states, actions, rewards, nexts, dones, log_probs =\
                self.memory.sample_transition()     
        # Discount the rewards 
        _rewards = self.calc_reward(rewards, dones)
        _rewards = torch.as_tensor(_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        _rewards = (_rewards - _rewards.mean()) / _rewards.std() 
        # Convert to tensors
        states = torch.as_tensor(states, device=self.device)
        _, state_values = self.network(states)
        advantage = _rewards - state_values.detach()
        
        # Calculating Actor loss
        self.network.optimizer.zero_grad()
        actor_loss = (-torch.stack(log_probs) * advantage)
        delta_loss = ((state_values - _rewards)**2)
        loss = (actor_loss + delta_loss).sum()
        loss.backward()
        self.network.optimizer.step()
        return [0, 0]   

    def train_on_step(self, state, action, reward, next_, next_action, log_probs):
        pass

    def clear_loss(self):
        self.actor_loss = []
        self.delta_loss = []
        pass

    def save_state(self):
        pass
    
    def calc_reward(self, rewards, dones):

        new_rewards = []
        _sum = 0
        for i in range(len(rewards)):
            r = rewards[-1+i]
            _sum *= self.gamma
            _sum += r
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        return new_rewards
    
    def store_transition(self, state, action, reward,
            _next, done, probs):
        self.memory.store_transition(state, action, reward,
                _next, done, probs=probs)
 
    def update_eps(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

