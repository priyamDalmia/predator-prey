import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np

class Network(nn.Module):
    def __init__(self, input_dims, output_dims, n_actions, lr):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.n_actions = n_actions
        # Parameters
        self.lr = lr
        self.fc1_dims = 256
        self.fc2_dims = 256

        # architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, 
                kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, 
                kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(300, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims , self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.conv2(x)
        x = self.flatten(x)
        breakpoint()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    def __init__(self, input_dims, output_dims, action_space, load_model, \
            memory, lr, gamma=0.95, epsilon=0.95, eps_end=0.01, \
            eps_dec=1e-4, batch_dims=64):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.memory = memory
        self.lr = lr 
        self.gamma = gamma
        
        # epsilon
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        
        # network
        self.network = Network(self.input_dims, self.output_dims,
                self.action_space, self.lr)
        self.network = (self.network).to(self.network.device)
    
    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        # T.tensor([np.array(observation)])
        state = T.tensor(np.array([observation]), dtype=T.float32).to("cuda:0")
        actions = self.network.forward(state)
        action = T.argmax(actions).item()
        return action

    def train_on_batch(self):
        if self.memory.counter < self.memory.batch_size:
            return None

        self.network.zero_grad()
        states, actions, rewards, next_, dones = self.memory.sample_batch()
        batch_idx = np.arange(self.memory.batch_size, dtype=np.int32)
        
        states = T.tensor(states).to(self.network.device)
        next_ = T.tensor(next_).to(self.network.device)
        rewards = T.tensor(rewards).to(self.network.device)
        
        values_t = self.network.forward(states)[batch_idx, actions]
        values_next = self.network.forward(next_)
        values_next[dones] = 0.0
        
        targets = rewards + self.gamma * T.max(values_next, dim=1)[0]

        loss = self.network.loss(targets, values_t).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end\
                else self.eps_end
        return loss.item()
    
    def update_epsilon(self):
        pass

    def store_transition(self, state, action, reward, next_, done):
        self.memory.store_transition(state, action, reward, next_, done)
