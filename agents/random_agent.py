import numpy as np

class RandomAgent():
    def __init__(self, input_dims, output_dims, action_space, replay_buffer):
        self.input_dims = input_dims
        self.outupt_dims = output_dims
        self.action_space = action_space
        self.replay_buffer = None

    def get_action(self, observation):
        return np.random.choice(self.action_space)
