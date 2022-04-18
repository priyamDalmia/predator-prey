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
            clayers = network_dims.clayers
            cl_dims = network_dims.cl_dims
            for c in range(nlayers):
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
        
        self.actor_layer = nn.Linear(nl_dims[-1], self.output_dims)
        self.critic_layer = nn.Linear(nl_dims[-1], self.output_dims)
        # Model Config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
    
    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
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
        action_dist = dist.Categorical(probs)
        action = action_dist.sample() 
        log_probs =  action_dist.log_prob(action)   
        return action.item(), log_probs

    def train_on_batch(self):
        self.network.optimizer.zero_grad()
        states, actions, rewards, nexts, dones, log_probs =\
                self.memory.sample_transition()     
        _rewards = self.calc_reward(rewards, dones)
        next_actions = np.append((actions[1:]), [0]) 
        # Convert to tensors
        states = torch.as_tensor(states, device=self.device)
        nexts = torch.as_tensor(nexts, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        _rewards = torch.as_tensor(_rewards, device=self.device).unsqueeze(-1)
        next_actions = torch.as_tensor(next_actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        _, state_values = self.network(states)
        _, next_values = self.network(nexts)
        
        _rewards = _rewards
        # Calculating Actor loss
        q_vals = torch.gather(state_values, dim=1, index=actions) 
        actor_loss = -1 * ((_rewards - q_vals) * torch.stack(log_probs))

        # Calculating Critic loss
        #dis_qvals = self.gamma * torch.gather(next_values, dim=1, 
         #       index=next_actions) * dones
        delta_loss = (_rewards - q_vals) ** 2
        #critic_loss = F.mse_loss(delta_loss, next_values)loss = (actor_loss + delta_loss).sum()
        loss = (actor_loss + delta_loss).mean()
        loss.backward()
        self.network.optimizer.step()
        return loss.item()

    def train_on_step(self, state, action, reward, next_, next_action, log_probs):
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

