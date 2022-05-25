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

class Memory:
    def __init__(self, num_agents,action_dims):
        self.num_agents = num_agents
        self.action_dims = action_dims
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(num_agents)]
        self.reward = []
        self.done = [[] for _ in range(num_agents)]

    def get(self):
        action = torch.tensor(self.actions)
        observations = self.observations

        pi = []
        for i in range(self.num_agents):
            pi.append(torch.cat(self.pi[i])).view(len(self.pi[i]), self.action_dim)
        
        reward = torch.tensor(self.reward)
        done = self.done
        
        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.num_agents)]
        self.reward = []
        self.done = [[] for _ in range(self.num_agents)]


class CriticNetwork(nn.Module):
    def __init__(self, 
            input_dims, 
            output_dims, 
            agent_ids, 
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
        
class NetworkActor(nn.Module):
    def __init__(self, 
            input_dims, 
            output_dims, 
            network_dims, 
            **kwargs):
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
            num_agents=1, agent_network={}, **kwargs):
        super(COMAAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr
        self.gamma = gamma 
        # Internal Memory
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_update_steps = 5
        self.memory = Memory()
        # Initialize the CAC Network 
        self.actors = []
        for i in range(self.num_agents):
            self.agent_network = agent_network
            actor = NetworkActor(input_dims, output_dims, 
                    self.agent_network)
            self.actors.append(actors)
        self.critic = NetworkCritic()
        self.critic_target = NetworkCritic()
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(), lr=self.lr, eps=1e-3) \
                for i in range(self.num_agents)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.count = 0

    def get_action(self, observation, _id=None):
        if not _id:
            raise("Wrong Trainer Initialzied. Use centralzied trainers and pass _id")
        idx = int(_id[-1])
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs = self.actor(observation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        return action.item(), 0

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
        actor_loss = torch.gather(log_probs, dim=1, index=actions) * advantage * -1
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
