import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from data.agent import BaseAgent
from data.helpers import dodict
import pdb

"""
An Implementation of the CounterFactual Multi-Agent Algorithm.
"""

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, agent_ids, 
            network_dims, **kwargs):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.agent_ids = agent_ids
        self.network_dims = network_dims 
        self.net = nn.ModuleList()

        # Network Architecture
        idim = self.input_dims[0]
        if len(self.input_dims)!=1:
            self.clayers = self.network_dims.clayers
            self.cl_dims = self.network_dims.cl_dims
            for c in range(self.clayers):
                self.net.append(
                        nn.Conv2d(in_channels=idim,
                            out_channels=self.cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = self.cl_dims[c]
            self.net.append(nn.Flatten())
            idim = self.cl_dims[-1] * \
                    ((self.input_dims[-1] - int(self.clayers))**2) + \
                    (self.output_dims + len(self.agent_ids))
        self.nlayers = self.network_dims.nlayers
        self.nl_dims = self.network_dims.nl_dims
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim, self.nl_dims[l]))
            idim = self.nl_dims[l]
        self.output_layer = nn.Linear(self.nl_dims[-1], self.output_dims)
        self.net.append(self.output_layer)
    
    def forward(self, x: tuple):
        inputs = x[0]
        for i, layer in enumerate(self.net):
            inputs = layer(inputs)
            if i == self.clayers:
                x0 = inputs 
                break
        inputs = torch.cat((x0, x[1], x[2]), dim=-1)
        for j, layer in enumerate(self.net[i+1:]):
            inputs = layer(inputs)
        return inputs
        
class CentCritic(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            action_space, agent_ids, lr=0.001, 
            gamma=0.95, betas=(0.9, 0.99), load_model=False, 
            eval_model=False, critic_network={}, **kwargs):
        super(CentCritic, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.agent_ids = agent_ids
        self.lr = lr
        self.gamma = gamma
        self.betas = betas 
        # Internal Memory
        self.obs_memory = []
        self.reward_memory = []
        self.action_memory = [[] for _ in range(len(self.agent_ids))]

        if self.load_model:
            pass
        self.critic_network = critic_network
        self.critic = CriticNetwork(input_dims, output_dims, agent_ids, 
                self.critic_network)
        self.critic_target = CriticNetwork(input_dims, output_dims, agent_ids,
                self.critic_network)
        

        self.optimizer = optim.Adam(self.critic.parameters(),
                lr=self.lr, betas=self.betas, eps=1e-3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)

    def train_step(self):
        states, actions, rewards = self.sample_transitions()
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        rewards_ = self.discount_rewards(rewards)
        rewards_ = torch.as_tensor(rewards_, dtype=torch.float32, device=self.device).unsqueeze(-1)
        len_ = len(states)
        ones = torch.ones((len_, 1), dtype=torch.int64).to(self.device)
        loss = []
        Q_values = {}
        for idx, _id in enumerate(self.agent_ids):
            actions_ = torch.as_tensor(actions[idx], device=self.device).unsqueeze(-1)
            actions_vec = F.one_hot(actions_, num_classes=self.output_dims).\
                    view(-1, self.output_dims).type(torch.float32)
            agents_vec = F.one_hot(ones*idx, num_classes=len(self.agent_ids)).\
                    view(len_, -1).type(torch.float32)
            values = self.critic((states, actions_vec, agents_vec))
            q_values = self.critic_target((states, actions_vec, agents_vec)).detach()
            Q_values[_id] = q_values
            # Backpropogating loss.
            q_taken = torch.gather(values, dim=1, index=actions_)
            critic_loss = torch.mean((rewards_ - q_taken)**2)
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.optimizer.step()
            # Store loss here.
        return critic_loss.item(), Q_values
            
    def store_transition(self, observations, actions, rewards):
        combined_obs = np.array([], dtype=np.int32)
        combined_obs = combined_obs.reshape(0, self.input_dims[1], self.input_dims[2])
        combined_rewards = 0
        for _id in self.agent_ids:
            combined_obs = np.vstack([combined_obs, observations[_id]])
            combined_rewards += rewards[_id]
        self.obs_memory.append(combined_obs)
        self.reward_memory.append(combined_rewards)
        for idx, _id in enumerate(self.agent_ids):
            self.action_memory[idx].append(actions[_id])
        
    def sample_transitions(self):
        states = np.stack(self.obs_memory, axis=0)
        actions = self.action_memory
        rewards = np.stack(self.reward_memory)
        self.obs_memory = []
        self.reward_memory = []
        self.action_memory = [[] for _ in range(len(self.agent_ids))]
        return states, actions, rewards

    def update_target_critic(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def save_checkpoint(self):
        pass

    def save_model(self, dir_):
        pass

    def get_action(self):
        pass

class NetworkActor(nn.Module):
    def __init__(self, input_dims, output_dims, 
            network_dims, **kwargs):
        super(NetworkActor, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        
        # Network Architecture
        idim = self.input_dims[0]
        if len(self.input_dims)!=1:
            self.clayers = self.network_dims.clayers
            self.cl_dims = self.network_dims.cl_dims
            for c in range(self.clayers):
                self.net.append(
                        nn.Conv2d(in_channels=idim,
                            out_channels=self.cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = self.cl_dims[c]
            self.net.append(nn.Flatten())
            idim = self.cl_dims[-1] * \
                    ((self.input_dims[-1]-int(self.clayers))**2)
        self.nlayers = network_dims.nlayers
        self.nl_dims = network_dims.nl_dims
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim , self.nl_dims[l]))
            idim = self.nl_dims[l]
        self.output_layer = nn.Linear(self.nl_dims[-1], self.output_dims)
        self.net.append(self.output_layer)

    def forward(self, x):
        for idx, layer in enumerate(self.net):
            x = layer(x)
        return F.softmax(x, dim=1)

class COMAAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            action_space, lr=0.01, gamma=0.95, 
            load_model=False, eval_model=False, 
            agent_network={}, **kwargs):
        super(COMAAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma 
        # Internal Memory
        self.action_memory = []
        self.prob_memory = []
        # Initialize the CAC Network 
        if self.load_model:
            try:
                model = torch.load(self.load_model)
                self.agent_network = dodict(model['agent_network'])
                self.network = NetworkActor(input_dims, output_dim, 
                        self.agent_network)
                self.network.load_state_dict(model['model_state_dict'])
                if eval_model:
                    self.network.eval()
                else:
                    self.network.train()
                print(f"Model Loaded:{_id} -> {self.load_model}")
            except:
                print(f"Load Failed:{_id} -> {self.load_model}")
        else:
            self.actor_network = agent_network
            self.actor = NetworkActor(input_dims, output_dims, 
                    self.actor_network)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor = self.actor.to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(),
                lr=self.lr, betas=(0.9, 0.99), eps=1e-3)

    def get_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs = self.actor(observation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        return action.item(), probs

    def train_step(self, q_values):
        """train_step.

        Args:
            Q_values: The Q values recived from the critic for the current trajectory.
            The Q-values correspond to this specific actor.
        """
        actions, probs = self.sample_transitions()
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        probs = torch.stack(probs).view(-1,4)
        baseline  = torch.sum(probs * q_values, dim=1).detach().reshape(-1,1)
        q_taken = torch.gather(q_values, dim=1, index=actions)
        # Get Values from the Critic Here
        # Calculate Advantage using the Centralized Critic.
        advantage = q_taken - baseline
        # Calculate and Backpropogate the Actor Loss.
        log_probs = torch.log(probs)
        actor_loss = torch.gather(log_probs, dim=1, index=actions) * advantage
        self.optimizer.zero_grad()
        loss = (actor_loss).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def store_transition(self, state, action, reward, 
            next_, done, probs, **kwargs):
        self.action_memory.append(action)
        self.prob_memory.append(probs)

    def sample_transitions(self):
        # Convert to numpy array and then return
        actions = np.stack(self.action_memory)
        probs = self.prob_memory
        # Return and reset memory.
        self.action_memory = []
        self.prob_memory = []
        return actions, probs

    def save_state(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.actor.state_dict()

    def save_model(self):
        torch.save({
            'model_state_dict': self.checkpoint,
            'loss': self.total_loss,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'network_dims': dict(self.agent_network),
            }, self.checkpoint_name)
        print(f"Model Saved: {self._id} -> {self.checkpoint_name}")
    
    def load_model(self):
        pass
