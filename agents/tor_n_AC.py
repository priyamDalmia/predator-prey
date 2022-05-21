import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from data.helpers import dodict
from data.agent import BaseAgent
import pdb

class NetworkActorCritic(nn.Module):
    def __init__(self, 
            input_dims,
            output_dims,
            network_dims,
            **kwargs):
        super(NetworkActorCritic, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.network_dims = network_dims
        self.nlayers = network_dims.nlayers
        self.nl_dims = network_dims.nl_dims
        self.clayers = network_dims.clayers
        self.cl_dims = network_dims.cl_dims
        self.net = nn.ModuleList()
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
                    nn.Linear(idim, self.nl_dims[l]))
            idim = self.nl_dims[l]
        self.actor_layer = nn.Linear(self.nl_dims[-1], self.output_dims)
        self.critic_layer = nn.Linear(self.nl_dims[-1], 1)

    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        logits = self.actor_layer(x)
        values = self.critic_layer(x)
        return F.softmax(logits, dim=1), values 

class AACAgent(BaseAgent):
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
        super(AACAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma
        self.initialize_memory(self.input_dims)
        self.device = torch.device('cuda:0'\
                if torch.cuda.is_available() else 'cpu')
        if load_model:
            checkpoint = torch.load(load_model)
            self.agent_network = dodict(checkpoint['agent_network'])
            self.network = NetworkActorCritic(
                    input_dims,
                    output_dims,
                    self.agent_network)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            if eval_model:
                self.network.eval()
            else:
                self.network.train()
        else:
            self.agent_network = agent_network
            self.network = NetworkActorCritic(
                    input_dims,
                    output_dims,
                    self.agent_network)
        self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.lr,
                betas=(0.9, 0.99),
                eps=1e-3)
        self.network = self.network.to(self.device)
    def get_action(self, observation):
        observation = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device)
        probs, _ = self.network(observation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def train_step(self):
        states, actions, rewards, dones = \
                self.sample_transition()
        if len(states) == 0:
            return 0
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        _rewards = self.discount_rewards(rewards, dones, states)
        _rewards = torch.as_tensor(_rewards, dtype=torch.float32, device=self.device)

        probs, state_values = self.network(states)
        prob_dist = dist.Categorical(probs)
        log_probs = prob_dist.log_prob(actions)
        advantage = _rewards - state_values.detach()

        self.optimizer.zero_grad()
        actor_loss = (-(log_probs) * advantage)
        delta_loss = ((state_values - _rewards) ** 2)
        loss = (actor_loss + delta_loss).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def discount_rewards(self, rewards, dones, states):
        if rewards[-1]==1 or rewards[-1]==1:
            _sum = rewards[-1]
        else:
            last_state = torch.as_tensor(
                    states[-1],
                    dtype=torch.float32,
                    device=self.device)
            _, value = self.network(last_state.unsqueeze(0))
            _sum = value.item()
        new_rewards = []
        new_rewards.append(_sum)
        rewards = np.flip(rewards)
        for i in range(1, len(rewards)):
            r = rewards[i]
            _sum *= self.gamma
            _sum += r
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        return new_rewards

    def initialize_memory(self, input_dims):
        self.mem_size = 2000
        self.mem_counter = 0
        self.states = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.actions = np.zeros((self.mem_size), dtype=np.int32)
        self.rewards = np.zeros((self.mem_size), dtype=np.float32)
        self.dones = np.zeros((self.mem_size), dtype=np.float32)
        self.probs = []

    def store_transition(self, state, action, reward, done):
        index = self.mem_counter % self.mem_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.done = 1 - int(done)
        self.mem_counter += 1

    def sample_transition(self):
        states = self.states[:self.mem_counter]
        actions = self.actions[:self.mem_counter]
        rewards = self.rewards[:self.mem_counter]
        dones = self.dones[:self.mem_counter]
        self.initialize_memory(self.input_dims)
        return states, actions, rewards, dones
    
    def save_state(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.network.state_dict()

    def save_model(self):
        torch.save({
            'model_state_dict': self.checkpoint,
            'loss': self.total_loss,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'agent_network': dict(self.agent_network),
            }, self.checkpoint_name)
        print(f"Model Saved: {self._id} -> {self.checkpoint_name}")
