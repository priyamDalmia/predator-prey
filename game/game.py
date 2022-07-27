import os 
import numpy as np
from typing import List
from game_utils import Action, Observation

class Environment:
    def __init__(self):
        pass

class GameHistory:
    def __init__(self):
        pass

class Game:
    def __init__(self, config: Config, env: Environment):
        self._env = env
        self._npred = config.npred
        self._nprey = config.nprey
        pass
    
    @property
    def npred(self):
        return self._npred
    
    @property
    def nprey(self):
        return self._nprey

    @property
    def state_space(self):
        pass

    @property
    def observation_space(self):
        pass
    
    @property
    def action_space(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass

    def is_terminal(self):
        pass
    
    def current_state(self, agent_id: str = None):
        if agent_id:
            # if agent id return specific obervation
            pass
        # else return Dict for all agents 
        pass

    def last_actions(self):
        pass

    def last_rewards(self):
        pass

    def last_messages(self):
        pass
    
    def render(self):
        pass


