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
        self.train: bool = False

    def get_action(self, observation=None):
        # TODO Fix action space
        return np.random.choice(4)

    def train_step(self):
        return dict(loss=None) 
    
    def store_transition(self, *args):
        pass

    def update_eps(self):
        pass

    def save_state(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

