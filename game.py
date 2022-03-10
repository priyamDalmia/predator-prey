import os
import logging
import numpy as np
import pdb

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

    def step(self, actions): 
        if len(actions) != self.map.units:
            raise ValueError("actions length does not match number of agents!")
        
        breakpoint()
        for idx, action in enumerate(actions):
            try:
                self.take_action(action, self.agents[idx])
                self.game_state.update_unit()
            except: 
                print("Invalid action {action_id} for current agent type: {self.agents_list[idx]}")

    def take_action(self, action_id, agent_id):
        breakpoint()
        if agent_id.startswith("predator"):
            pos_x, pos_y = self.predators[agent_id].get_position()
            ACTIONS_PRED[action_id](pos_x, pos_y, self.size)
        elif agent_id.startswith("pery"):
            pos_x, pos_y = self.preys[agent_id].get_position()
            ACTIONS_PREY[action_id](pos_x, pos_y, self.size)
        else:
            pass

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
                    self.agents_list.append(f"predator_{i}")
                    break
        return 0

    def render(self, mode="human"):
        # renders obstacels to O
        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
        for i in self.predators.values():
            (x, y) = i.get_position()
            gmap[x][y] = "T"

        for i in self.preys.values():
            (x, y) = i.get_position()
            gmap[x][y] = "P"
        
        #breakpoint()
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]    
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
    
    def update_unit(self):
        pass

    def check_collision(self, pos_x, pos_y):
        if np.sum(self.state, axis=0)[pos_x, pos_y]:
            return 1
        return 0

    def state(self):
        pass

