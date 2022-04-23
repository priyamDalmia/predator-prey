import os
import sys
import copy
import numpy as np
import pandas as pd
from game.game import Game
from data.helpers import dodict
from argparse import parser
from data.agent import BaseAgent
from data.agent import RandomAgent

config = {}

class Evaluate():
    def __init__(self, env, config, *args, **kwargs):
        self.env = env
        self.config = config
        
        # Initialize Agnets (Load Agents)
        self.agents_ids = env.agent_ids
        self.agents = self.initialize_agents()

        # Bookeeping
        self.steps_avg = 0
        self.rewards_avg = 0
        self.loss_avg = 0

    def run_episodes(self):
        steps_hist = []
        reward_hist = []
        
        for ep in range(self.config.episodes):
            pass
        pass

    def initialize_agents(self):
        pass

if __name__ == "__main__":
    pass

