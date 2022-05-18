from abc import ABC, abstractmethod
from data.agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def __init__(self, _id, input_dims, output_dims, action_space,
            **kwargs):
        super(RandomAgent, self).__init__(_id)
        self.input_dims = input_dims
        self.outupt_dims = output_dims
        self.action_space = action_space
        self.memory = None
        self.memory_n = None

    def get_action(self, observation):
        return np.random.choice(self.action_space), 0

    def train_step(self):
        return 0
    
    def store_transition(self, *args):
        pass

    def update_eps(self, *args):
        pass

    def save_state(self, *args):
        pass

    def save_model(self, *args):
        pass

    def load_model(self, filename):
        pass

