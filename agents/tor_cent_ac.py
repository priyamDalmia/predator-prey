import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from data.agent import BaseAgent
import pdb

class NetworkCritic(nn.Module):
    def __init__(self, input_dims, output_dims, memory=None,
            network_dims={}, lr=0.001, gamma=0.95, **kwargs):
        super(NetworkCritic, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        # Bookeeping
        self.checkpoint = None
        self.checkpoint_name = None
        self.total_loss = 0
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
                        nn.Conv2d(in_channels=idim,
                            out_channels=cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = cl_dims[c]
            self.net.append(nn.Flatten())
            ### Must be modified if the newtwork paramters change
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim , nl_dims[l]))
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

    def train_step(self):
        # Combined States and Rewards for all agents.
        states, rewards = self.memory.sample_transition()
        # Calculate Dsicounted Rewards
        _rewards = self.discount_rewards(rewards)
        _rewards = torch.as_tensor(_rewards, device=self.device)
        states = torch.as_tensor(states, device=self.device)
        state_values = self.forward(states)
        # Calculating the net advanatge for each state.
        advantage = _rewards - state_values.detach()
        # Caluclating Critic loss
        delta_loss = ((state_values - _rewards)**2)
        loss = (delta_loss).mean()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        self.total_loss += loss_value
        return state_values.detach() , loss_value
    
    def discount_rewards(self, rewards):
        new_rewards = []
        _sum = 0
        for i in range(len(rewards)):
            r = rewards[-1+i]
            _sum *= self.gamma 
            _sum += r
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        return new_rewards
    
    def store_transition(self, observation, rewards):
        combined_obs = np.array([], dtype=np.int32)
        combined_obs = combined_obs.reshape(0, self.input_dims[1], self.input_dims[2])
        combined_rewards = 0
        for _id in observation.keys():
            if _id.startswith("predator"):
                combined_obs = np.vstack([combined_obs, observation[_id]])
                combined_rewards += rewards[_id]
        self.memory.store_transition(combined_obs, combined_rewards)

    def save_checkpoint(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.net.state_dict()

    def save_model(self):
        model_name = f"trained-policies/multi/{self.checkpoint_name}"
        torch.save({
            'model_state_dict':self.checkpoint,
            'loss': self.total_loss,
            'input_dims':self.input_dims,
            'output_dims':self.output_dims,
            'network_dims': dict(self.network_dims),
            }, model_name)
        print(f"Critic Model Saved")

class NetworkActor(nn.Module):
    def __init__(self, input_dims, output_dims, action_space,
            lr, network_dims, **kwargs):
        super(NetworkActor, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.lr = lr
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        idim = self.input_dims[0]
        # Network Layers
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
            ### Must be modified if the newtwork paramters change
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim , nl_dims[l]))
            idim = nl_dims[l]
        self.actor_layer = nn.Linear(nl_dims[-1], self.output_dims)
        
    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        logits = self.actor_layer(x)
        return F.softmax(logits, dim=1)

class CACAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            action_space, memory=None, lr=0.01, gamma=0.95,
            load_model=False, actor_network=None, critic=None,
            **kwargs):
        super(CACAgent, self).__init__(_id)
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
        self.actor_network = actor_network
        if not isinstance(critic, NetworkCritic):
            print(f"WARNING: Critic not of class NetworkCritic")
        self.critic = critic
        if self.load_model:
            # Load Models for both Actor and Critic Here!
            pass
        else:
            self.actor_network = NetworkActor(input_dims, output_dims, action_space,
                    lr, self.actor_network)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor_network = self.actor_network.to(self.device)
        self.optimizer = optim.Adam(self.actor_network.parameters(),
                lr=self.lr, betas=(0.9, 0.99), eps=1e-3)

    def get_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs = self.actor_network(observation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        return action.item(), log_probs

    def train_step(self, state_values):
        states, actions, rewards, nexts, dones, log_probs = \
                self.memory.sample_transition()
        # Discount the rewards 
        _rewards = self.discount_rewards(rewards, dones)
        _rewards = torch.as_tensor(_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)

        states = torch.as_tensor(states, device=self.device)
        # Get Values from the Critic Here
        breakpoint()
        # Calculate Advantage using the Centralized Critic.
        advantage = _rewards - state_values
        # Calculate and Backpropogate the Actor Loss.
        self.optimizer.zero_grad()
        actor_loss = (-torch.stack(log_probs) * advantage)
        loss = (actor_loss).mean()
        loss.backward()
        self.optimizer.step()
        return [loss.item(), 0]

    def discount_rewards(self, rewards, dones):
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
            next_, done, probs, **kwargs):
        if self.memory:
            self.memory.store_transition(state, action, reward,
                    next_, done, probs=probs)

    def save_state(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.actor_network.state_dict()

    def save_model(self):
        model_name = f"trained-policies/multi/{self.checkpoint_name}"
        torch.save({
            'model_state_dict': self.checkpoint,
            'loss': self.total_loss,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'network_dims': dict(self.network_dims),
            }, model_name)
    
    def load_model(self):
        pass
