from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch 

class BaseAgent(ABC):
    def __init__(self, _id):
        self._id = _id
        self.input_dims = None
        self.output_dims = None
        self.action_space = None
        self.lr = None
        self.gamma = None
        self.memory = None
        self.epsilon = None
        self.device = None
        self.checkpoint = None
        self.checkpoint_name = None
        self.network = None
        self.total_loss = 0

    @abstractmethod
    def get_action(self, observation):
        """
        Takes an observation as input and returns a action
        
        Args:
           observation: current observation of the agent.

        """
        pass
    
    @abstractmethod
    def store_transition(self):
        pass

    @abstractmethod
    def train_step(self):
        pass
    
    def save_state(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint = self.network.state_dict()

    def save_model(self, *args):
        torch.save({
             'model_state_dict': self.checkpoint,
             'loss': self.total_loss,
             'input_dims': self.input_dims,
             'output_dims': self.output_dims,
             'agent_network': dict(self.agent_network),
             }, self.checkpoint_name)
        print(f"Model Saved {self._id} | {model_name}")

    def load_model(self, filename):
        pass
    
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

    def clear_loss(self):
        pass

    def update_eps(self):
        pass
