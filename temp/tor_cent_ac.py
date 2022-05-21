import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from data.agent import BaseAgent
import pdb

"""
An implementation of the CounterFactual Multi-Agent Algorithm.
"""

class NetworkCritic(nn.Module):
    """NetworkCritic.
    """
    def __init__(self, observation_dims, output_dims, action_space, 
            pred_ids, memory=None, network_dims={}, lr=0.001, 
            gamma=0.95, **kwargs):
        super(NetworkCritic, self).__init__()
        self.observation_dims = observation_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.pred_ids = pred_ids
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        # Bookeeping
        self.checkpoint = None
        self.checkpoint_name = None
        self.total_loss = 0
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        # Memory for Actions
            
        self.action_mem = [[] for _ in range(len(self.pred_ids))]
        # Network Layers
        idim = self.observation_dims[0]
        if len(self.observation_dims) != 1:
            # Convolutional Layers
            self.clayers = self.network_dims.clayers
            cl_dims = self.network_dims.cl_dims
            for c in range(self.clayers):
                self.net.append(
                        nn.Conv2d(in_channels=idim,
                            out_channels=cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = cl_dims[c]
            self.net.append(nn.Flatten())
            # Add or extend on to a new layer here. 
            # + Action Vector (for each agent)
            # + Id Vector
            ### Must be modified if the newtwork paramters change
            idim = cl_dims[-1] * \
                    ((self.observation_dims[-1]-int(self.clayers))**2)
        self.nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        # Expanding idim for join actions and agent ids
        idim = idim + (self.output_dims) +(len(self.pred_ids))
        for l in range(self.nlayers):
            self.net.append(
                    nn.Linear(idim , nl_dims[l]))
            idim = nl_dims[l]
        self.critic_layer = nn.Linear(nl_dims[-1], self.output_dims)
        self.net.append(self.critic_layer)
        self.optimizer = optim.Adam(self.parameters(),
                lr=self.lr, betas=(0.9, 0.99), eps=1e-3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
    def forward(self, inputs):
        """forward.
        Outputs the Q valeus for the given observation, joint actions and agent _id.
        Args:
            inputs: [obesrvation_U, action_U, agent_id]
        """
        observations = inputs[0]
        for i in range(self.clayers):
            layer = self.net[i]
            x = layer(observations)
            observations = x
        i+=1
        x = self.net[i](x)
        try:
            x = torch.cat((x, inputs[1].type(torch.float32), inputs[2].type(torch.float32)), dim=-1)
        except Exception as e:
            print(e)
            breakpoint()
        for j in range(self.nlayers):
            layer = self.net[j+i+1]
            x = layer(x)
        # Final Layer
        x = self.net[-1](x)
        return x

    def train_step(self):
        """train_step.
        Returns a dictionary of Q values corresponding to each agent. This is an 
        effectie way to address the credit assignment problem.
        """
        loss = 0
        # Combined States and Rewards for all agents.
        states, rewards = self.memory.sample_transition()
        batch_len = len(states)
        # Calculate Dsicounted Rewards
        _rewards = self.discount_rewards(rewards)
        _rewards = torch.as_tensor(_rewards, device=self.device).unsqueeze(-1)
        Q_values = {}
        for idx, _id in enumerate(self.pred_ids):    
            states = torch.as_tensor(states, device=self.device)
            actions = torch.as_tensor(self.action_mem[idx], device=self.device).unsqueeze(-1)
            actions_vec = F.one_hot(actions, num_classes=len(self.action_space)).view(-1, 4)
            agent_id = F.one_hot(torch.ones((batch_len,1), dtype=torch.int64)*idx, num_classes=len(self.pred_ids))\
                .view(batch_len, -1).to(self.device)
            q_values = self.forward((states, actions_vec, agent_id))
            Q_values[_id] = q_values.detach()

            # Not using a target network here!
            q_taken = torch.gather(q_values, dim=1, index=actions)
            critic_loss = torch.mean((_rewards - q_taken)**2)
            self.optimizer.zero_grad()
            critic_loss.backward()
            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()
        self.clear_memory()
        return loss, Q_values

    def discount_rewards(self, rewards):
        new_rewards = []
        _sum = 0
        rewards = np.flip(rewards)
        for i in range(len(rewards)):
            r = rewards[i]
            _sum *= self.gamma 
            _sum += r
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        return new_rewards
     
    def store_transition(self, observation, actions, rewards):
        combined_obs = np.array([], dtype=np.int32)
        combined_obs = combined_obs.reshape(0, self.observation_dims[1], self.observation_dims[2])
        combined_rewards = 0
        for _id in observation.keys():
            if _id.startswith("predator"):
                combined_obs = np.vstack([combined_obs, observation[_id]])
                combined_rewards += rewards[_id]
        for indx, _id in enumerate(self.pred_ids):
            self.action_mem[indx].append(actions[_id])
        self.memory.store_transition(combined_obs, combined_rewards)

    def clear_memory(self):
        self.action_mem = [[] for _ in range(len(self.pred_ids))]

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
            lr=0.01, gamma=0.95, agent_network={},
            load_model=False, eval_model=False, **kwargs):
        super(COMAAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.load_model = load_model
        self.lr = lr
        self.gamma = gamma 
        # Memory
        self.action_memory = actions
        self.prob_memory = log_probs
        # Initialize the CAC Network 
        if self.load_model:
            try:
                
                model = torch.load(self.load_model)
                self.agent_network = dodict(model['agent_network'])
                self.network = NetworkActorCritic(input_dims, output_dims, action_space,
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
            self.actor_network = actor_network
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
        breakpoint()
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
        breakpoint()
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
        torch.save({
            'model_state_dict': self.checkpoint,
            'loss': self.total_loss,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'network_dims': dict(self.actor_network),
            }, self.checkpoint_name)
        print(f"Model Saved: {self._id} -> {self.checkpoint_name}")
    
    def load_model(self):
        pass
