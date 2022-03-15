import os
import logging
import numpy as np
#import pdb

from data.entites import Predator, Prey
from data.common import ACTIONS_PRED, ACTIONS_PREY
from game_state import GameState

class Game():
    def __init__(self, size, npred, nprey, nobstacles=0, win_size=3):
        self.size = size
        self.win_size = win_size
        self.pad_width = int(self.win_size/2)
        self.npred = npred
        self.nprey = nprey
        self.nobstacles = nobstacles 
        self.game_state = GameState(self.size, self.win_size)
        self.agent_ids = []
        self.predators = {}
        self.preys = {}        
        
        # Populate predators and prey characters 
        self.create_predators(self.npred)
        self.create_prey(self.nprey)
        self.last_actions = []

    def step(self, actions): 
        if len(actions) != self.game_state.units:
            raise ValueError("actions length does not match number of agents!")
        
        for idx, action in enumerate(actions):
            try:
                self.take_action(str(action), self.agent_ids[idx])
            except: 
                print(f"Invalid action {action} for current agent type: {self.agent_ids[idx]}")        
                
        self.last_actions = actions 
        return 0, self.is_done()
   
    def observe(self, agent_id):
        channel_id = self.agent_ids.index(agent_id)
        if agent_id.startswith("predator"):
            pos_x, pos_y = self.predators[agent_id].get_position()
            agent_obs = self.game_state.state[:, 
                    pos_x-self.pad_width:pos_x+self.pad_width+1, 
                    pos_y-self.pad_width:pos_y+self.pad_width+1]
        elif agent_id.startswith("prey"):
            pos_x, pos_y = self.preys[agent_id].get_position()
            agent_obs = self.game_state.state[:, 
                    pos_x-self.pad_width:pos_x+self.pad_width+1, 
                    pos_y-self.pad_width:pos_y+self.pad_width+1]

        return agent_obs
        
    def reset(self):
        # Destroy agents and create new objects 
        self.predators = {}
        self.preys = {}
        self.agent_ids = []
        # Clear game_state object 
        self.game_state.reset()
        self.last_actions = []
        # Re-populate all characters 
        self.create_predators(self.npred)
        self.create_prey(self.nprey)
        
        return self.is_done()

    def is_done(self):
        if self.game_state.units > 0:
            return False
        return True

    def take_action(self, action_id, agent_id):
        if agent_id.startswith("predator"):
            pos_x, pos_y = self.predators[agent_id].get_position()
            pos_x, pos_y = ACTIONS_PRED[action_id](pos_x, pos_y, self.size, self.win_size)
            self.predators[agent_id].set_position(pos_x, pos_y)
        elif agent_id.startswith("prey"):
            pos_x, pos_y = self.preys[agent_id].get_position()
            pos_x, pos_y = ACTIONS_PREY[action_id](pos_x, pos_y, self.size, self.win_size)
            self.preys[agent_id].set_position(pos_x, pos_y)
        try:
            self.game_state.update_unit(self.agent_ids.index(agent_id)+1, pos_x, pos_y)
        except Exception as e:
            print(e)
            print(f"Unit update failed {action_id} , {agent_id}")

    def create_predators(self, npred):
        predators = {}
        low = self.pad_width
        high = self.size + self.pad_width
        for i in range(npred):
            while True:
                pos_x, pos_y = np.random.randint(low=low, high=high, size=2)
                pred = Predator(i, pos_x, pos_y, 1)
                if self.game_state.add_unit(pred):
                    self.predators[f"predator_{i}"] = pred
                    self.agent_ids.append(f"predator_{i}")
                    break
        return 0 

    def create_prey(self, nprey):
        preys = {}
        low = self.pad_width
        high = self.size + self.pad_width
        for i in range(nprey):
            while True:
                pos_x, pos_y = np.random.randint(low=low, high=high, size=2)
                prey = Prey(i, pos_x, pos_y, 1)
                if self.game_state.add_unit(prey):
                    self.preys[f"prey_{i}"] = prey
                    self.agent_ids.append(f"prey_{i}")
                    break
        return 0

    def render(self, mode="human"):
        # renders obstacels to O
        # Function to adjust (x[0], x[1]) by +y)
        ADJ = lambda x, y: (x[0]-y, x[1]-y)

        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
        for agent in self.predators.values():
            (x, y) = ADJ(agent.get_position(), self.pad_width)
            gmap[x][y] = "T"

        for agent in self.preys.values():
            (x, y) = ADJ(agent.get_position(), self.pad_width)
            gmap[x][y] = "D"
        
        
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]    
        LINE_CLEAR = '\x1b[2k'
        for _ in range(10):  print(end=LINE_CLEAR)
        print(np.matrix(gmap))
        #state = self.game_state.state
        #gmap = np.zeros((self.size, self.size), dtype=np.int32)
        #breakpoint()


