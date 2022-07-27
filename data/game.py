import os 
import numpy as np
from data.config import Config
from data.game_utils import ObservationSpace, ActionSpace, Observation, Action
from typing import Dict

class Environment:
    def __init__(self, state_space: Tuple, actors: Dict, metadata: Dict):
        self._state_space = state_space        
        self._actors = actors
        self._metadata = metadata
        # build observation and action_spaces 
        self.observation_spaces = {}
        self.action_spaces = {}
        for actor_id, actor in self._actors:
            self.observation_spaces[actor_id] =\
                    ObservationSpace(actor.observation_space)
            self.action_spaces[actor_id] =\
                    ActionSpace(actor.action_space)
    
    @property
    def state_space(self):
        return self._state_space

    @property
    def metadata(self):
        return self._metadata

    def observation_space(self, actor_id = None) -> Dict:
        if actor_id:
            return self.observation_spaces[actor_id]
        return self.observation_spaces

    def action_space(self, actor_id = None) -> Dict:
        if actor_id:
            return self.action_spaces[actor_id]
        return self.action_spaces

class GameHistory:
    def __init__(self):
        pass

class Game:
    def __init__(self, config: Config, env: Environment):
        self.config = config
        self._env = env
        pass
    
    def state_space(self):
        self._env.state_space

    def observation_space(self):
        pass

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


