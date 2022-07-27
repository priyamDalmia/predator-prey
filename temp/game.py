import os 
import sys
import random
import numpy as np
from game.utils import Action, Observation 
from game.config import Config

class Game:
    def __init__(self, config: Config):
        self.config = config
        self.size = config.size
        self.nprey = config.nprey
        self.npred = config.npred

        # TODO add atrritbutes , __setter__, __getter__
        self.state_space = []
        self.action_space = []

    def reset(self):
        pass

    # profile step functions 
    def step(self, actions: LIST, meta: LIST = None):
        current_rewards = {}
        action_group = []
        # take standard action here
        for , unit in enumerate(action_group):
            if isinstance(unit, PREDATOR):

                # get action 
                # check for collision 
                 # if team 
                 # no action
                 # if prey
                 # capture and reward
                pass
            else:
                pass


        # take reward distribution step

        # take message passing steps
        # seperate message planes for boardcasting 
        # or personal messages in observation

        # log game info in written - messages + rewards + actions before capture

        pass

    def is_terminal(self,):
        pass

    def legal_actions(self, agent: Agent = None, mask: bool = False):
        pass
    
    def last_rewards(self):
        pass
