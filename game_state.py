import os
import numpy as np

class GameState():
    def __init__(self, size, window_size):
        self.size = size
        self.window_size = window_size
        self.obstacles = 0
        self.units = 0
        
        # Creating a base layer : channel
        # Stack base layers to form the Game state array.
        self.pad_width = int(self.window_size / 2)
        self.channel = np.pad(np.zeros((size, size), dtype=np.int32), pad_width=self.pad_width, constant_values=-1)
        self.state = np.expand_dims(self.channel, axis=0).copy()

    def add_obstacles(self):
        pass
    
    def get_observation(self, channel_id, agent_id):
        '''
        ! window_size can only be an odd number
        '''
        breakpoint()
        observation = self.state
        pass

    def add_unit(self, unit):
        pos_x, pos_y = unit.get_position()
        if not self.check_collision(pos_x, pos_y):
            new_channel = self.channel.copy()
            new_channel[pos_x, pos_y] = 1
            self.state = np.vstack((self.state, np.expand_dims(new_channel, axis=0)))
            # Use vastack and expand_dims to stack arrays
            self.units += 1
            return 1
        return 0
    
    def update_unit(self, unit_id, pos_x, pos_y):
        self.state[unit_id, :, :] = self.channel.copy()
        self.state[unit_id, pos_x, pos_y] = 1 
        

    def check_collision(self, pos_x, pos_y):
        if np.sum(self.state, axis=0)[pos_x, pos_y]:
            return 1
        return 0

    def state(self):
        pass

    def reset(self):
        # Creating a base layer : channel
        # Stack base layers to form the Game state array.
        self.channel = np.pad(np.zeros((self.size, self.size), dtype=np.int32), pad_width=self.pad_width, constant_values=-1)
        self.state = np.expand_dims(self.channel, axis=0).copy()
        self.units = 0

