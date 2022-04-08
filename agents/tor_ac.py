import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data.agent import BaseAgent

class Network(nn.Module):
    def __init__(self, input_dims, output_dims, action_space, **kwargs):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space 

    def forward(self, inputs):
        return x
    
class ActorAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, action_space, 
            load_model, memory=None, lr=0.001, gamma=0.95,
            epsilon=0.95, epsilon_end=0.01, epsilon=1e-4, **kwargs):
        super(ActorAgent, self).__init__(_id, name="")
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.load_model = load_model

    def get_action(self, observation):
        pass

    def train_on_batch(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
