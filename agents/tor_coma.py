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
    def __init__(self, num_agents, state_size):
        self.num_agents = num_agents
        self.buffer_size = 1000
        self.state_size = state_size
        self.counter = 0
        self.states = {}
        self.actions = {}
        self.rewards = {}
        self.dones = {}
        self.probs = {}
        self.critic_states = {}
        for i in range(num_agents):
            _id = f"predator_{i}"
            self.states[_id] = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
            self.actions[_id] = np.zeros((self.buffer_size), dtype=np.int32)
            self.rewards[_id] = np.zeros((self.buffer_size), dtype=np.float32)
            self.dones[_id] = np.zeros((self.buffer_size), dtype=np.float32)
            self.probs[_id] = []
        critic_size = (self.num_agents*self.state_size[0], self.state_size[1], self.state_size[2])
        self.critic_states = np.zeros((self.buffer_size, *critic_size), dtype=np.float32)
        self.critic_actions = np.zeros((self.buffer_size, self.num_agents), dtype=np.int32)
    
    def get_agent_transition(self, _id):
        states = self.states[_id][:self.counter]
        actions = self.actions[_id][:self.counter]
        rewards = self.rewards[_id][:self.counter]
        dones = self.dones[_id][:self.counter]
        probs = self.probs[_id][:self.counter]
        if _id == f"predator_{self.num_agents-1}":
            self.counter = 0
            for i in range(self.num_agents):
                self.probs[_id] = []
        return states, actions, rewards, dones, probs

    def store_reward_done(self, _id, reward, done):
        index = self.counter % self.buffer_size
        self.rewards[_id][index] = reward
        self.dones[_id][index] = done

    def store_action_prob(self, _id, state, action, prob):
        index = self.counter % self.buffer_size
        self.states[_id][index] = state
        self.actions[_id][index] = action
        self.probs[_id].append(prob)

    def store_critic_transition(self, observations, actions):
        index = self.counter % self.buffer_size
        self.critic_states[index] = observations
        self.critic_actions[index] = actions
        self.counter += 1

    def get_critic_transition(self):
        states = self.critic_states[:self.counter]
        actions = self.critic_actions[:self.counter]
        return states, actions

class NetworkCritic(nn.Module):
    def __init__(self, 
            input_dims, 
            output_dims, 
            num_agents, 
            network_dims, 
            **kwargs):
        super(NetworkCritic, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_agents = num_agents
        self.network_dims = network_dims 
        self.net = nn.ModuleList()

        # Network Architecture
        idim = self.input_dims[0] * self.num_agents
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
                    (self.num_agents + 1)
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
    def __init__(self, 
            _id, 
            input_dims, 
            output_dims, 
            action_space, 
            lr, 
            gamma, 
            num_agents, 
            agent_network, 
            critic_network,
            nsteps=30,
            **kwargs):
        super(COMAAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr
        self.gamma = gamma 
        self.num_agents = num_agents
        self.agent_network = agent_network
        self.critic_network = critic_network
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_update_steps =15
        # Internal Memory
        self.memory = Memory(self.num_agents, self.input_dims)
        # Initialize the CAC Network 
        self.actors = []
        actor = NetworkActor(
                input_dims, 
                output_dims, 
                self.agent_network)
        actor = actor.to(self.device)
        for i in range(self.num_agents):
            self.actors.append(actor)
        self.critic = NetworkCritic(
                input_dims, 
                output_dims, 
                num_agents, 
                self.critic_network)
        self.critic = self.critic.to(self.device)
        self.critic_target = NetworkCritic(
                input_dims,
                output_dims,
                num_agents,
                self.critic_network)
        self.critic_target.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(), lr=self.lr, eps=1e-3) \
                for i in range(self.num_agents)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.count = 0

    def get_action(self, observation, _id=None):
        if not _id:
            raise("Wrong Trainer Initialzied. Use centralzied trainers and pass _id")
        idx = int(_id[-1])
        inputs = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        # IF possible; concatenate Agents with shared _ids.
        probs = self.actors[idx](inputs.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        action = action.item()
        # Store Obervation, Action and Probs
        self.memory.store_action_prob(_id, observation, action, probs.detach())
        return action, 0

    def train_step(self):
        """train_step.

        """
        critic_obs, critic_actions = self.memory.get_critic_transition()
        if len(critic_obs) == 0:
            return 0  
        for i in range(self.num_agents):
            agent_id = f"predator_{i}"
            states, actions, rewards, dones, probs = self.memory.get_agent_transition(agent_id)
            actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
            critic_input = self.build_critic_input(critic_obs, critic_actions, i)
            q_value = self.critic_target(critic_input).detach()
            
            actor_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            probs = self.actors[i].forward(actor_states)
            baseline  = torch.sum(probs * q_value, dim=1).detach().reshape(-1,1)
            q_taken = torch.gather(q_value, dim=1, index=actions)
            advantage = q_taken - baseline
            log_probs = torch.log(probs)
            actor_loss = torch.gather(log_probs, dim=1, index=actions) * advantage * -1
            actor_loss = actor_loss.mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            self.actor_optimizers[i].step()
            # Use states Values to create baseline
            rewards = self.discount_rewards(rewards, dones, q_taken[-1])
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
            Q = self.critic(critic_input)
            Q_taken = torch.gather(Q, dim=1, index=actions)
            critic_loss = torch.mean((rewards - Q_taken)**2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            self.critic_optimizer.step()
        self.count += 1
        if self.count == self.target_update_steps:
            self.count = 0
            self.critic_target.load_state_dict(self.critic.state_dict())
        return 0

    def discount_rewards(self, rewards, dones, q_taken):
        if rewards[-1]==1 or rewards[-1]==-1:
            _sum = rewards[-1]
        else:
            _sum = q_taken.item()
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


    def build_critic_input(self, states, actions, idx):
        nlen = states.shape[0]
        ids = torch.ones(nlen, 1, device=self.device) * idx
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return (states, actions, ids) 

    def store_transition(self, _id, reward, done):
        self.memory.store_reward_done(_id, reward, done)
    
    def store_critic_transitions(self, observations, actions):
        combined_obs = np.array([], dtype=np.int32)
        combined_obs = combined_obs.reshape(0, self.input_dims[1], self.input_dims[2])
        combined_actions = np.zeros((1, self.num_agents), dtype=np.int32)
        for _id in observations:
            if _id.startswith("predator"):
                index = int(_id[-1])
                combined_obs = np.vstack([combined_obs, observations[_id]])
                combined_actions[0, index] = actions[_id]
        self.memory.store_critic_transition(combined_obs, combined_actions)

    def save_state(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.actors[0].state_dict()

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
