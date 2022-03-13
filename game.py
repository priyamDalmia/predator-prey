import os
import logging
import numpy as np
#import pdb

from data.entites import Predator, Prey
from data.common import ACTIONS_PRED, ACTIONS_PREY

class Game():
    def __init__(self, size, npred, nprey, nobstacles=0):
        self.size = size
        self.npred = npred
        self.nprey = nprey
        self.nobstacles = nobstacles 
        self.game_state = GameState(self.size)
        self.agents_list = []
        self.predators = {}
        self.preys = {}        
        # Populate predaotrs and prey chracters 
        self.create_predators(self.npred)
        self.create_prey(self.nprey)
        self.last_actions = []

    def step(self, actions): 
        if len(actions) != self.game_state.units:
            raise ValueError("actions length does not match number of agents!")
        

        for idx, action in enumerate(actions):
            try:
                self.take_action(str(action), self.agents_list[idx])
            except: 
                print(f"Invalid action {action} for current agent type: {self.agents_list[idx]}")        
                
        self.last_actions = actions 
        return 0, self.is_done()
   
    def observe(self, agent):
        
        pass

    def reset(self):
        # Destroy agents and create new objects 
        self.predators = {}
        self.preys = {}
        self.agents_list = []
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
            pos_x, pos_y = ACTIONS_PRED[action_id](pos_x, pos_y, self.size)
            self.predators[agent_id].set_position(pos_x, pos_y)
        elif agent_id.startswith("prey"):
            pos_x, pos_y = self.preys[agent_id].get_position()
            pos_x, pos_y = ACTIONS_PREY[action_id](pos_x, pos_y, self.size)
            self.preys[agent_id].set_position(pos_x, pos_y)
        try:
            self.game_state.update_unit(self.agents_list.index(agent_id)+1, pos_x, pos_y)
        except Exception as e:
            print(e)
            print(f"Unit update failed {action_id} , {agent_id}")

    def create_predators(self, npred):
        predators = {}
        for i in range(npred):
            while True:
                pos_x, pos_y = np.random.randint(0, high=self.size, size=2)
                pred = Predator(i, pos_x, pos_y, 1)
                if self.game_state.add_unit(pred):
                    self.predators[f"predator_{i}"] = pred
                    self.agents_list.append(f"predator_{i}")
                    break
        return 0 
    def create_prey(self, nprey):
        preys = {}
        for i in range(nprey):
            while True:
                pos_x, pos_y = np.random.randint(0, high=self.size, size=2)
                prey = Prey(i, pos_x, pos_y, 1)
                if self.game_state.add_unit(prey):
                    self.preys[f"prey_{i}"] = prey
                    self.agents_list.append(f"prey_{i}")
                    break
        return 0

    def render(self, mode="human"):
        # renders obstacels to O
        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
        for agent in self.predators.values():
            (x, y) = agent.get_position()
            gmap[x][y] = "T"

        for agent in self.preys.values():
            (x, y) = agent.get_position()
            gmap[x][y] = "D"
        
        
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]    
        LINE_CLEAR = '\x1b[2k'
        for _ in range(10):  print(end=LINE_CLEAR)
        print(np.matrix(gmap))
        #state = self.game_state.state
        #gmap = np.zeros((self.size, self.size), dtype=np.int32)
        #breakpoint()

class GameState():
    def __init__(self, size):
        self.size = size
        self.obstacles = 0
        self.units = 0
        self.state = np.zeros(shape=(1, size, size), dtype=np.int32)
        self.channel = np.zeros(shape=(1, size, size), dtype=np.int32)

    def add_obstacles(self):
        pass

    def add_unit(self, unit):
        (pos_x, pos_y) = unit.get_position()
        if not self.check_collision(pos_x, pos_y):
            channel = np.zeros(shape=(1, self.size, self.size), dtype=np.int32)
            channel[0, pos_x, pos_y] = 1
            self.state = np.vstack((self.state, channel))
            self.units += 1
            return 1
        return 0
    
    def update_unit(self, unit_id, pos_x, pos_y):
        self.state[unit_id, :, :] = np.zeros((self.size, self.size))
        self.state[unit_id, pos_x, pos_y] = 1 
        

    def check_collision(self, pos_x, pos_y):
        if np.sum(self.state, axis=0)[pos_x, pos_y]:
            return 1
        return 0

    def state(self):
        pass

    def reset(self):
        self.units = 0
        self.state = np.zeros(shape=(1, self.size, self.size), dtype=np.int32)
