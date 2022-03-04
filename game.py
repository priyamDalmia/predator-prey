import os
import logging
import numpy as np

from data.entites import Predator, Prey

class Game():
    def __init__(self, size, npred, nprey, nobstacles=0):
        self.size = size
        self.npred = npred
        self.nprey = nprey
        self.nobstacles = nobstacles 
        self.game_state = Gamestate(self.size)
        
        self.predators = self.create_predators(npred)
        self.preys = self.create_prey(nprey)
    
    def step(self, actions):
        pass

    def create_predators(self, npred):
        predators = {}
        for i in range(npred):
            while True:
                pos_x, pos_y = np.random.randint(0, high=self.size, size=2)
                pred = Predator(i, pos_x, pos_y, 1)
                if self.game_state.add_unit(pred):
                    predators[f"predator_{i}"] = pred
                    break
        return predators

    def create_prey(self, nprey):
        preys = {}
        for i in range(nprey):
            while True:
                pos_x, pos_y = np.random.randint(0, high=self.size, size=2)
                prey = Prey(i, pos_x, pos_y, 1)
                if self.game_state.add_unit(prey):
                    preys[f"prey_{i}"] = prey
                    break
        return preys

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




class Gamestate():
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
   

    def check_collision(self, pos_x, pos_y):
        if np.sum(self.state, axis=0)[pos_x, pos_y]:
            return 1
        return 0

    def state(self):
        pass

