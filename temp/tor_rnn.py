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
            lr, network_dims, num_agents, **kwargs):
        super(NetworkActorCritic, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.lr = lr
        self.network_dims = network_dims
        self.net = nn.ModuleList()
        self.recurrent = True
        self.num_agents = num_agents
        self.hidden_state = {}
        self.last_hidden = {}
        try: 
            self.rnn_layers = self.network_dims.rnn_layers
        except:
            self.rnn_layers = 2
        # Network Layers
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Convolutional Layers
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
            #### Must be modified if the network parameters change
            idim = self.cl_dims[-1] * \
                    ((self.input_dims[-1]-int(self.clayers))**2)
        self.nlayers = network_dims.nlayers
        self.nl_dims = network_dims.nl_dims
        for l in range(self.nlayers):
            if l < 1: 
                self.net.append(
                    nn.RNN(idim, self.nl_dims[l], self.rnn_layers))
            else:
                self.net.append(
                    nn.Linear(idim, self.nl_dims[l]))
            idim = self.nl_dims[l]
        self.clear_hidden_states()
        self.actor_layer = nn.Linear(self.nl_dims[-1], self.output_dims)
        self.critic_layer = nn.Linear(self.nl_dims[-1], 1)
    
    def forward(self, inputs, **kwargs):
        _id = kwargs["_id"]
        for i, layer in enumerate(self.net):
            inputs = layer(inputs)
            if i == self.clayers:
                break
        self.last_hidden[_id] = self.hidden_state[_id]
        for j, layer in enumerate(self.net[i+1:]):
            if j == 0:
                inputs, self.hidden_state[_id] = layer(inputs.unsqueeze(0), self.last_hidden[_id])
                inputs = inputs.squeeze(0)
            else:
                inputs = layer(inputs)
        logits = self.actor_layer(inputs)
        values = self.critic_layer(inputs)
        return F.softmax(logits, dim=1), values, self.last_hidden[_id].detach()
    
    def forward_train(self, inputs, hidden_state):
        for i, layer in enumerate(self.net):
            inputs = layer(inputs)
            if i == self.clayers:
                break
        for j, layer in enumerate(self.net[i+1:]):
            if j == 0:
                inputs, _ = layer(inputs.unsqueeze(0), hidden_state)
                inputs = inputs.squeeze(0)
            else:
                inputs = layer(inputs)
        logits = self.actor_layer(inputs)
        values = self.critic_layer(inputs)
        return F.softmax(logits, dim=1), values
    
    def clear_hidden_states(self):
        for i in range(self.num_agents):
            self.hidden_state[f"predator_{i}"] = torch.zeros((self.rnn_layers, 1, self.nl_dims[0]))
            self.last_hidden[f"predator_{i}"] = None

class AACAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, 
            action_space, memory=None, lr=0.01, gamma=0.95,
            load_model=False, eval_model=False, agent_network={},
            num_agents = None, **kwargs):
        super(AACAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model 
        self.lr = lr
        self.gamma = gamma 
        self.memory = memory
        self.memory_n = {}
        self.num_agents = num_agents
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
                        lr, self.agent_network, self.num_agents)
        
        self.optimizer = optim.Adam(self.network.parameters(),
                lr = self.lr, betas=(0.9, 0.99), eps=1e-3)
        self.deivce = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(self.device)
        torch.autograd.set_detect_anomaly(True)
    
    def get_action(self, observation, _id = None):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                device=self.device)
        probs, values, hidden_state = self.network(observation.unsqueeze(0), _id=_id)
        action_dist = dist.Categorical(probs)
        action = action_dist.sample() 
        return action.item(), hidden_state
    
    def get_raw_output(self, observation, _id=None):
        with torch.no_grad():
            observation = torch.as_tensor(observation, dtype=torch.float32,
                    device=self.device)
            probs, values = self.network(observation.unsqueeze(0))
        return probs, values
    
    def train_step(self, _id):
        states, actions, rewards, dones, hidden_states =\
                self.memory_n[_id].sample_transition()
        if len(states) == 0:
            return 0
        # Discount the rewards
        _rewards = self.discount_rewards(rewards, dones, states, hidden_states)
        _rewards = torch.as_tensor(_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        probs_list = []
        state_values = []
        for i in range(len(states)):
            hidden_state = torch.as_tensor(hidden_states[i], dtype=torch.float32, device=self.device)
            last_state = torch.as_tensor(states[i], dtype=torch.float32, device=self.device)
            probs, value = self.network.forward_train(last_state.unsqueeze(0), hidden_state=hidden_state)
            probs_list.append(probs)
            state_values.append(value.detach())
        # Dirty
        state_values = torch.stack(state_values).view(len(states),1)
        probs = torch.stack(probs_list)
        log_probs = torch.gather(torch.log(probs).squeeze(1), 1, index=actions.unsqueeze(-1))

        advantage = _rewards - state_values
        # Calculating Loss and Backpropogating the error.
        self.optimizer.zero_grad()
        actor_loss = (-(log_probs) * advantage)
        delta_loss = ((state_values - _rewards)**2)
        loss = (actor_loss + delta_loss).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def store_transition(self, _id, state, action, reward,
            done, hidden_state):
        if self.memory_n[_id]:
            if state.size == 0:
                return
            self.memory_n[_id].store_transition(state, action, reward,
                done, hidden_state=hidden_state)
    
    def clear_loss(self):
        self.actor_loss = []
        self.delta_loss = []

    def clear_hidden_states(self):
        self.network.clear_hidden_states()

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

    def discount_rewards(self, rewards, dones, states, hidden_states):
        if rewards[-1]==1 or rewards[-1]==-1:
            _sum = rewards[-1]
        else:
            last_state = torch.as_tensor(states[-1], dtype=torch.float32,
                    device=self.device)
            hidden_state = torch.as_tensor(hidden_states[-1], dtype=torch.float32,
                    device=self.device)
            p, value = self.network.forward_train(last_state.unsqueeze(0), hidden_state = hidden_state)
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


