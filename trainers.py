import os
import sys
import numpy as np

from data.config import TrainerConfig

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.env_name = config.env_name
        self.env_config = config.env_config
        # use this to build game objects

    def initialize_agents(self):
        pass

    def initialize_game(self):
        pass

    def initialize_logger(self):
        pass

    def initialize_training(self):
        pass
    
    def run_service(self):
        pass

    def run_training(self):
        pass

    def run_episode(self):
        pass

    def make_log(self):
        pass

    def compile_results(self):
        pass

    


