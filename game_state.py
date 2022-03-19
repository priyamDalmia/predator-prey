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
        self.predator_pos = []
        self.prey_pos = []
        # stacking channels along the axis 0 to get the game state.
        self.channel = np.pad(np.zeros((size, size), dtype=np.int32), pad_width=pad_width, constant_values=-1)
        self.state = np.expand_dims(self.channel, axis=0).copy()

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
            self.predator_pos.append((pos_x, pos_y))
        elif _id.startswith("prey"):
            agent = Prey(_id, pos_x, pos_y, 1)
            self.prey_pos.append((pos_x, pos_y))

        return agent, (pos_x, pos_y)
    
    def observe(self, _id, channel_id, agent, position):
        pos_x, pos_y = position
        observation = self.state[:, 
                pos_x-self.pad_width:pos_x+self.pad_width+1, 
                pos_y-self.pad_width:pos_y+self.pad_width+1]
        return observation

    def update_units_1(self, next_preds, next_preys, rewards):
        '''
        A stochastic parallel unit update function. Predators and preys have equal probabilites to move and 
        react first. 
        '''
        choice = random.choice([0,1])
        breakpoint()
        if choice == 0:
            pass
        try:
            for idx, position in enumerate(next_preds):
                if idx < self.npreds:
                    if position in self.prey_pos:
                        rewards[idx] = 10
                    self.state[idx+1, :, :] = self.channel.copy()
                    self.state[idx+1, position[0], position[1]] = 1
                #self.predator_pos[idx] = position
                else:
                # Only Predators can move over a prey; since the very goal is to consume the prey
                # For prey movements; collision must be detected and movement reversed.
                    if position in self.predator_pos:
                        next_positions[idx] = self.prey_pos[idx-self.npreds]
                
                    if position in next_positions[:self.npreds]:
                        rewards[idx] = -10
                        while True:
                            pos_x, pos_y = np.random.randint(low=self.low, high=self.high, size=2)
                            if not np.sum(self.state, axis=0)[pos_x, pos_y]:
                                break
                    self.state[idx+1, :, :] = self.channel.copy()
                    self.state[idx+1, position[0], position[1]] = 1
        except Exception as e:
            print(e)
        # update game state predator positions and prey positions
        self.predator_pos = next_positions[:self.npreds]
        self.prey_pos = next_positions[self.npreds:]
        return rewards, next_positions

    def update_units(self, next_preds, next_preys, rewards):
        '''
        All the units act at the same time!
        '''
        breakpoint()

        pred_ids = next_preds.keys()
        random.shuffle(pred_ids)
        for _id in pred_ids:
            position = next_preds[_id]
            preys = list(next_preys.values())
            if position in preys:
                next_prey[f"prey_{preys.index(position)}"] = 0
                rewards[_id] += 10
                self.state[int(_id[-1])+1, :, :] = self.channel.copy()
                self.state[int(_id[-1])+1, position[0], position[1]] = 1
                break
            if position in self.prey_pos:
                prey_idx = self.prey_pos.index(position)
                next_preys[f'prey_{prey_idx}'] = 0
                rewards[_id] += 10
                self.state[int(_id[-1])+1, :, :] = self.channel.copy()
                self.state[int(_id[-1])+1, position[0], position[1]] = 1
            self.predator_pos[int(_id[-1])] = position            
       
        for _id, position in next_preys.items():
            if position:
                self.state[int(_id[-1])+1+self.npreds, :, :] = self.channel.copy()
                self.state[int(_id[-1])+1+self.npreds, position[0], position[1]] = 1
            else:
                reward[_id] += -10
                # The prey was caputred!
                self.state[int(_id[-1])+1+self.npreds, :, :] = self.channel.copy()
                self.state[int(_id[-1])+1+self.npreds, position[0], position[1]] = 0
                del next_preds[_id]
        self.prey_pos = list(next_preys.values())
        
        next_preds.update(next_preys)
        return rewards, next_preds
