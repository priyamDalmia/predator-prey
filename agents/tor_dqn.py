import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data.agent import BaseAgent

import pdb

class NetworkLinear(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, lr, **kwargs):
        super(NetworkLinear, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space 
        self.lr = lr 
        self.fc1_dims = 256
        self.fc2_dims = 256
        # Network layers
        self.layer1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.layer2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.layer3 = nn.Linear(self.fc2_dims, self.output_dims)
        # Model config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        x = F.relu(self.layer1(inputs))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class NetworkConv(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, lr, **kwargs):
        super(NetworkConv, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space 
        self.lr = lr
        self.fc1_dims = 256
        self.fc2_dims = 256
        # Network layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12,
                kernel_size=1, stride=1)  
        # Add code for dynamic layer creation here.
        # Use try except statements.
        conv_out = (12 * input_dims[-1] * input_dims[-1])
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(conv_out, self.fc1_dims)
        self.layer2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.layer3 = nn.Linear(self.fc2_dims, self.output_dims)
        # model config
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DQNAgent(BaseAgent):
    def  __init__(self, _id, input_dims, output_dims, action_space, 
             load_model, memory=None, lr=0.0001, gamma=0.95,
            epsilon=0.90, epsilon_end=0.01, epsilon_dec=1e-4, **kwargs):
        """
        An DQN Agent with target network and a Prioritized Exp replay.
        Args:
            _id: game agent _id
            input_dims: tuple
            output_dims: int
            action_space: list 
            load_model: bool
            memory: ReplayBuffer 
            lr: float 
            gamma: float
            epsilon: float
            epsilon_end: float 
            epsilon_dec: float

        """
        super(DQNAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model
        # training variables  
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        # Add code for dynamic optimizer and loss functions.
        self.network = None
        self.target_network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.load_model:
            pass

        # Create new network if self.load model is false.
        if len(input_dims) == 1:
            self.network = NetworkLinear(self.input_dims, 
                    self.output_dims, self.action_space, self.lr)
            self.target_network = NetworkLinear(self.input_dims,
                    self.output_dims, self.action_space, self.lr)
        else:
            self.network = NetworkConv(self.input_dims, 
                    self.output_dims, self.action_space, self.lr)
            self.target_network = NetworkConv(self.input_dims, self.output_dims,
                    self.action_space, self.lr)
        self.network = self.network.to(self.device)
        self.target_network = self.target_network.to(self.device)

    @torch.no_grad()
    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space), 0
        # convert observation to tensor of shape : [batch_size, (input)]
        inputs = torch.as_tensor(observation, dtype=torch.float32, 
                device=self.device)
        values = self.network(inputs.unsqueeze(0))
        action = torch.argmax(values)
        return action.item(), action.item()

    def train_step(self):
        if self.memory.counter < self.memory.batch_size:
            return None
        
        # samples states, action, rewards and .. from the sample buffer 
        states, actions, rewards, nexts, dones = self.memory.sample_batch()
        
        # Convert to tensors
        states = torch.as_tensor(states, device=self.device)
        nexts = torch.as_tensor(nexts, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, device=self.device)

        # get values of states and nexts
        # Use critic network here!
        state_values = self.network(states)
        y_pred = torch.gather(state_values, dim=1, index=actions)
        
        next_values = self.network(nexts)
        targets = rewards + self.gamma * torch.max(next_values, dim=1)[0]
        
        self.network.optimizer.zero_grad()
        loss = self.network.loss(y_pred, (targets.unsqueeze(-1)))
        loss.backward()
        self.network.optimizer.step()

        # create the target values for the loss propogation
        return loss.item()
   
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        pass

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec
        return self.epsilon 

    def store_transition(self, state, action, reward, next_, done, *args, **kwargs):
        self.memory.store_transition(state, action, reward, next_, done)
        pass

    def save_state(self):
        return self.network.state_dict()

    def load_state(self, state):
        return self.network.load_state_dict(state)

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
