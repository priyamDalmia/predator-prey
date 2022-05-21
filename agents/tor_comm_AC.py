import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from data.helplers import dodict
from data.agents import BaseAgent
import pdb

class NetworkCommAC(nn.Module):
    def __init__(self,
            input_dims,
            output_dims,
            action_space,
            network_dims, 
            num_agents,
            **kwargs):
        super(NetworkCommAC, self).__init__()
        self.input_dims = input_dims
        
