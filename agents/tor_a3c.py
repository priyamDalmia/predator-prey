import torch
import torch.nn as nn
import torch.nn.funcitonal as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from data.agent import BaseAgent
import pdb

class NetworkActorCritic(nn.Module):
    def __init__():
        super().__init__()
        pass

    def forward(self, inputs):
        pass

class A3CAgent(BaseAgent):
    def __init__(self):
        super(A3CAgent, self).__init__()
        pass

    def get_action(self):
        pass

    def train_on_batch(self):
        pass
    
    def store_transition(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

    def update_eps(self):
        pass

