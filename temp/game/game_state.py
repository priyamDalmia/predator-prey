import os
import numpy as np
import random
import traceback
from data.entites import Predator, Prey
# random, agents.pread and preys, numpy
from data.common import ACTIONS



class GameState():
    def __init__(self, size, npreds, npreys, window_size, pad_width):
        self.size = size
        self.window_size = window_size
        self.pad_width = pad_width
        self.npreds = npreds
        self.npreys = npreys
        self.units = 0        
        self.low = self.pad_width
        self.high = self.size + self.pad_width

        # stacking channels along the axis 0 to get the game state.
        self.channel = np.pad(np.zeros((size, size), dtype=np.int32), pad_width=pad_width, constant_values=1)
        self.state = np.expand_dims(self.channel, axis=0).copy()
        self.channel = np.pad(np.zeros((size, size), dtype=np.int32), pad_width=pad_width, constant_values=0)

    def add_unit(self, _id) -> object:
        while True:
            pos_x, pos_y = np.random.randint(low=self.low, high=self.high, size=2)
            if not np.sum(self.state, axis=0)[pos_x, pos_y]:
                new_channel = self.channel.copy()
                new_channel[pos_x, pos_y] = 1
                self.state = np.vstack((self.state, np.expand_dims(new_channel, axis=0)))
                break
        self.units+=1
        agent = None
        if _id.startswith("predator"):
            agent = Predator(_id, pos_x, pos_y, 1)
        elif _id.startswith("prey"):
            agent = Prey(_id, pos_x, pos_y, 1)
        return agent, (pos_x, pos_y)
    
    def observe(self, _id, channel_id, pos_x, pos_y):
        observation = self.state[:, 
                pos_x-self.pad_width:pos_x+self.pad_width+1, 
                pos_y-self.pad_width:pos_y+self.pad_width+1]
        return observation

    def update_unit(self, idx, position):
        self.state[idx, :, :] = self.channel.copy()
        self.state[idx, position[0], position[1]] = 1

    def predator_collision(self, pos_x, pos_y):
        return np.sum(self.state[1:self.npreds+1, :, :], axis=0)[pos_x, pos_y]

    def prey_collision(self, pos_x, pos_y):
        return np.sum(self.state[self.npreds+1:, :,:], axis=0)[pos_x, pos_y]


