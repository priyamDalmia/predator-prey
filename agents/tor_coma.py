import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.Functional as F
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
                        nn.Conv2D(in_channels=idim,
                            out_channels=self.cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = self.cl_dims[c]
            self.net.append(nn.Flatten())
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1] - int(self.clayers))**2) + \
                    (self.output_dims + len(self.agent_ids))
        self.nlayers = self.network_dims.nlayers
        self.nl_dims = self.network_dims.nl_dims
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim, self.nl_dims[l]))
            idim = self.nl_dims[l]
        self.output_layer = nn.Linear(nl_dims[-1], self.output_dims)
        self.net.append(self.output_layer)
    
    def forward(self, x: tuple):
        inputs = x[0]
        for i, layer in enumerate(self.net):
            inputs = layer(inputs)
            if i == self.clayers:
                x0 = inputs 
                break
        inputs = torch.cat((x0, x[1], x[2]), dim=-1)
        for j, layer in enumerate(self.net[i:]):
            inputs = layer(inputs)
        return inputs
        

class CentCritic(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, agent_ids, 
            network_dims, lr=0.005, gamma=0.95, **kwargs):
        super(BaseAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.agent_ids = agent_ids
        self.network_dims = network_dims
        self.lr = lr
        self.gamma = gamma
        self.obs_memory = []
        self.reward_memory = []
        self.action_memory = [[] for _ in range(len(self.agent_ids))]

        self.critic = CriticNetwork()
        self.critic_target = CriticNetwork()

        self.optimizer = optim.Adam(self.critic.parameters(),
                lr=self.lr, betas=self.betas, eps=1e-3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_step(self):
        states, actions, rewards = self.sample_transitions()
        states = torch.as_tensor(states, device=self.device)
        rewards_ = self.discount_rewards(rewards)
        rewards_ = torch.as_tensor(rewards_, device=self.device).unsqueeze(-1)
        len_ = len(states)
        ones = torch.ones((len_, 1), dtype=torch.int64)
        loss = []
        Q_values = {}
        for idx, _id in enumerate(self.agent_ids):
            actions = torch.as_tensor(actions[idx], device=self.device).unsqueeze(-1)
            actions_vec = F.one_hot(actions, num_classes=self.output_dims).\
                    view(-1, self.output_dims).type(torch.float64)
            agents_vec = F.one_hot(ones*idx, num_classes=self.output_dims).\
                    view(len_, -1).type(torch.float64)
            values = self.critic.forward((states, actions_vec, agents_vec))

            # Backpropogating loss.
            q_taken = torch.gather(values, dim=1, index=actions)
            critic_loss = torch.mean((rewards_ - q_taken)**2)
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.optimizer.step()
            
            # Store loss here.

            # Get Q values from Target network
            q_values = self.critic_target().detach()
            Q_values[_id] = q_values
        return loss, Q_values
            
        pass

    def store_transition(self, observations, actions, rewards):
        combined_obs = np.array([], dtype=np.int32)
        combined_obs = combined_obs.reshape(0, self.input_dims[0], self.output_dims[0])
        combined_rewards = 0
        for _id in self.agent_ids:
            combined_obs = np.vstack([combined_obs, observations[_id]])
            combined_rewards += rewards[_id]
        self.obs_memory.append(combined_obs)
        self.reward_memory.append(combined_reward)
        for idx, _id in enumerate(self.pred_ids):
            self.action_memory[idx].append(actions[_id])
        
    def sample_transitions(self):
        breakpoint()
        self.obs_memory = []
        self.reward_memory = []
        self.action_memory = []
        return states, actions, rewards

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
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        self.nlayers = network_dims.nlayers
        self.nl_dims = network_dims.nl_dims
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim , nl_dims[l]))
            idim = nl_dims[l]
        self.output_layer = nn.Linear(nl_dims[-1], self.output_dims)
        self.net.append(self.output_layer)

    def forward(self, x):
        for idx, layer in enumerate(self.net):
            x = layer(x)
        return F.softmax(x, dim=1)

class COMAAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            network_dims={} lr=0.01, gamma=0.95, load_model=False,
            **kwargs):
        super(COMAAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma 
        # Memory
        self.action_memory = actions
        self.prob_memory = log_probs
        # Bookeeping
        self.checkpoint = None
        self.checkpoint_name = None
        # Initialize the CAC Network 
        self.actor_network = actor_network
        if self.load_model:
            # Load Models for both Actor and Critic Here!
            pass
        else:
            self.actor = NetworkActor(input_dims, output_dims, action_space,
                    lr, self.actor_network)
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
        log_probs = torch.stack(log_probs).view(-1,4)
        baseline  = torch.sum(log_probs * q_values, dim=1).detach().reshape(-1,1)
        q_taken = torch.gather(q_values, dim=1, index=actions)
        # Get Values from the Critic Here
        # Calculate Advantage using the Centralized Critic.
        advantage = q_taken - baseline
        # Calculate and Backpropogate the Actor Loss.
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
        breakpoint()
        # Convert to numpy array and then return
        actions = self.action_memory
        probs = self.prob_meomry

    def save_state(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.actor.state_dict()

    def save_model(self):
        model_name = f"trained-policies/multi/{self.checkpoint_name}"
        torch.save({
            'model_state_dict': self.checkpoint,
            'loss': self.total_loss,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'network_dims': dict(self.actor_network),
            }, model_name)
        print(f"Model Saved: {model_name}")
    
    def load_model(self):
        pass
