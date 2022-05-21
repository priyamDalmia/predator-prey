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
    def __init__(self, input_dims, output_dims, action_space,
            lr, network_dims,  **kwargs):
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
                        nn.Conv2d(in_channels=idim,
                            out_channels=cl_dims[c],
                            kernel_size=(2,2),
                            stride=1))
                idim = cl_dims[c]
            self.net.append(nn.Flatten())
            #### Must be modified if the network parameters change
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        nlayers = network_dims.nlayers
        nl_dims = network_dims.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            idim = nl_dims[l]
        self.actor_layer = nn.Linear(nl_dims[-1], self.output_dims)
        self.critic_layer = nn.Linear(nl_dims[-1], 1)
    
    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x
        logits = self.actor_layer(x)
        values = self.critic_layer(x)
        return F.softmax(logits, dim=1), values

class AACAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            action_space, memory=None, lr=0.01, gamma=0.95,
            load_model=False, eval_model=False, agent_network={},
            **kwargs):
        super(AACAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model 
        self.lr = lr
        self.gamma = gamma 
        self.memory = memory
        self.memory_n = {}
        # Initialize the AC network
        if self.load_model:
            try:
                checkpoint = torch.load(self.load_model)
                self.agent_network = dodict(checkpoint['agent_network'])
                self.network = NetworkActorCritic(input_dims, output_dims, action_space,
                        lr, self.agent_network)
                self.network.load_state_dict(checkpoint['model_state_dict'])
                if eval_model:
                    self.network.eval()
                else:
                    self.network.train()
                print(f"Model Loaded:{_id} -> {self.load_model}")
            except Exception as e:
                print(e)
                print(f"Load Failed:{_id} -> {self.load_model}")
        else:
            self.agent_network = agent_network
            self.network = NetworkActorCritic(input_dims, output_dims, action_space,
                        lr, self.agent_network)
        
        self.optimizer = optim.Adam(self.network.parameters(),
                lr = self.lr, betas=(0.9, 0.99), eps=1e-3)
        self.deivce = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(self.device)
        torch.autograd.set_detect_anomaly(True)

    def get_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs, values = self.network(observation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = action_dist.sample() 
        log_probs =  action_dist.log_prob(action)
        return action.item(), 0
    
    def get_raw_output(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(observation, dtype=torch.float32,
                    device=self.device)
            probs, values = self.network(observation.unsqueeze(0))
        return probs, values
    
    def train_step(self, _id):
        states, actions, rewards, _ , dones, _ =\
                self.memory_n[_id].sample_transition()
        if len(states) == 0:
            return 0
        # Discount the rewards
        _rewards = self.discount_rewards(rewards, dones, states)
        _rewards = torch.as_tensor(_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        actions = torch.as_tensor(actions)
        # Convert to tensors
        states = torch.as_tensor(states, device=self.device)
        probs, state_values = self.network(states)
        prob_dist = dist.Categorical(probs)
        log_probs = prob_dist.log_prob(actions)
        advantage = _rewards - state_values.detach()
        # Calculating Loss and Backpropogating the error.
        self.optimizer.zero_grad()
        actor_loss = (-(log_probs) * advantage)
        delta_loss = ((state_values - _rewards)**2)
        loss = (actor_loss + delta_loss).mean()
        loss.backward()
        nn.utils.clip_grad_value_(self.network.parameters(), clip_value=0.5)
        self.optimizer.step()
        return loss.item()

    def store_transition(self, _id, state, action, reward,
            _next, done, probs):
        if self.memory_n[_id]:
            if state.size == 0:
                return
            self.memory_n[_id].store_transition(state, action, reward,
                _next, done, probs=probs)
    
    def clear_loss(self):
        self.actor_loss = []
        self.delta_loss = []

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

    def discount_rewards(self, rewards, dones, states):
        if rewards[-1]==1 or rewards[-1]==-1:
            _sum = rewards[-1]
        else:
            last_state = torch.as_tensor(states[-1], dtype=torch.float32,
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


