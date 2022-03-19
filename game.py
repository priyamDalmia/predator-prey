import os
import logging
import numpy as np
#import pdb
import time

from data.entites import Predator, Prey
from data.common import ACTIONS
from game_state import GameState


class Game():
    def __init__(self, args):
        self.size = args.size
        self.npredators = args.npred
        self.npreys = args.nprey
        self.window_size = args.win_size
        self.pad_width = int(args.win_size/2)
        self.game_state = None

        # Game managment variables
        self.units = (self.npredators + self.npreys + 1)
        self.action_space = [i for i in range(4)]
        self.observation_space = np.zeros((self.units, self.size, self.size), dtype=np.int32)
        self.last_action = None
        self.last_rewards = None
        # Reset the environment object (basically recreates a new GameState object.)
        self.reset()   

    
    def reset(self) -> dict:
        self.agents = {}
        self.agent_ids = []
        # Create a new GameState Object
        self.game_state = GameState(self.size, self.npredators, self.npreys, self.window_size, self.pad_width)

        # Populate Predators
        for i in range(self.npredators):
            _id = f"predator_{i}"
            agent_obj , position = self.game_state.add_unit(_id)
            self.agents[_id] = (len(self.agent_ids)+1, agent_obj, position)
            self.agent_ids.append(_id)
        # Populate Preys 
        for i in range(self.npreys):
            _id = f"prey_{i}"
            agent_obj, position = self.game_state.add_unit(_id)
            self.agents[_id] = (len(self.agent_ids)+1, agent_obj, position)
            self.agent_ids.append(_id)       
        
        # returns the initial observation dict.
        return self.get_observation()

    def step(self, actions:dict) -> tuple:
        # takes an action dict and updates the underlying game state.
        # step over the action list and get new positions.
        # ! Modify for any changes in the reward structure here !
     
        if len(actions)!=len(self.agent_ids):
            raise print("Error: action sequence of incorrect length!")

        next_positions = []
        #rewards = [0 for i in range(len(self.agents))]
        idx = 0
        next_preds = {}
        next_preys = {}
        rewards = {} 
        self.render()
        try:
            for _id, action in actions.items():
                agent = self.agents[_id][1]
                rewards[_id] = 0
                pos_x, pos_y = self.agents[_id][2]
                new_pos = ACTIONS[str(action)](pos_x, pos_y, self.size, self.pad_width)
                if new_pos == (pos_x, pos_y):
                    rewards[_id] = -1
                if idx < self.npredators:
                    next_preds[_id] = new_pos
                else:
                    if new_pos in next_positions:
                        next_preys[_id] = (pos_x, pos_y)
                        break
                    next_preys[_id] = new_pos
                next_positions.append(new_pos)
                idx+=1
        except KeyError:
            print(f"Error: invalid action:{action} for agent:{_id}")
        
        print(actions)
        print(next_preys)

        rewards, next_positions = self.game_state.update_units(next_preds, next_preys, rewards)

        breakpoint()

        if sum(rewards) > 0:
            self.toggle()

        for index, _id in enumerate(self.agent_ids):
            (self.agents[_id][1]).set_position(*next_positions[index])
        
        self.last_action = actions
        self.last_rewards = rewards
        done = False
        info = {}
        return rewards, self.get_observation, done, info

    def get_observation(self) -> dict:
        observation = {}
        for _id, value in self.agents.items():
            observation[_id] = self.game_state.observe(_id, *value)
        self.observation = {}
        return observation

    def render(self, mode="human"):
    
        adjust = lambda x, y: (x[0]-y, x[1]-y)
        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
        for idx, position in enumerate(self.game_state.predator_pos):
            (x, y) = adjust(position, self.pad_width)
            gmap[x][y]  = f"T{idx}"
        for idx, position in enumerate(self.game_state.prey_pos):
            (x, y) = adjust(position, self.pad_width)
            gmap[x][y]  = f"D{idx}"
        
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]
        print(np.matrix(gmap))

    def toggle(self):
        self.render()
        breakpoint()


