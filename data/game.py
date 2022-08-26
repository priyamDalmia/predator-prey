import os 
import numpy as np
from data.config import Config
from data.game_utils import ObservationSpace, ActionSpace
from typing import Dict, Tuple

# TODO action to string for each
# TODO build abstract methods and look for abstract variables 

class Environment:
    def __init__(self, state_space: Tuple, actors: Dict, metadata: Dict):
        self._state_space = state_space        
        self._actors = actors
        self._metadata = metadata
        # build observation and action_spaces 
        self.observation_spaces = {}
        self.action_spaces = {}
        for actor_id, actor in self._actors.items():
            self.observation_spaces[actor_id] =\
                    ObservationSpace(actor.observation_space)
            self.action_spaces[actor_id] =\
                    ActionSpace(actor.action_space)
    @property
    def actor_ids(self):
        return list(self._actors.keys())

    @property
    def state_space(self):
        return self._state_space
    
    @property
    def metadata(self):
        return self._metadata

    @property
    def is_terminal(self):
        # TODO wrong - use math . prod instead. 
        return bool(np.prod(list(self._dones.values())))

    @property
    def steps(self):
        return self.steps

    def observation_space(self, actor_id = None) -> Dict:
        if actor_id:
            return self.observation_spaces[actor_id]
        return self.observation_spaces

    def action_space(self, actor_id = None) -> Dict:
        if actor_id:
            return self.action_spaces[actor_id]
        return self.action_spaces
    
    def actor_observations(self, actor_id = None):
        if actor_id:
            return self.observations[actor_id]
        return self.observations

    def render(self) -> np.array:
        # TODO iteratue over acotr sassigning Ids and numbers to positions
        lims = self.pad_width
        render_array = self._env_state[lims-1:-lims+1, lims-1:-lims+1, 0].copy() 
        for i in range(1, self.NUM_CHANNELS):
            render_array += self._env_state[lims-1:-lims+1, lims-1:-lims+1, i] 
        return render_array

class GameHistory:
    def __init__(self):
        pass

class Game:
    def __init__(self, config: Config, env: Environment):
        self.config = config
        self._env = env
        self.history = GameHistory()
        pass
    
    @property
    def agent_ids(self):
        return self._env.actor_ids
    
    @property
    def state_space(self):
        return self._env.state_space

    @property
    def timestep(self):
        return self._env.steps

    def observation_space(self, actor_id = None):
        if actor_id:
            return self._env.observation_space(actor_id)
        return self._env.observation_space()

    def action_space(self, actor_id = None):
        if actor_id:
            self._env.observation_space(actor_id)
        return self._env.action_space(actor_id)

    def reset(self):
        return self._env.reset()

    def step(self, actions: Dict):
        return self._env.step(actions)

    def get_observations(self, actor_id = None):
        if actor_id:
            self._env.actor_observations()
        return self._env.actor_observations(actor_id)

    def is_terminal(self):
        return self._env.is_terminal
    
    def current_state(self, agent_id = None):
        if agent_id:
            # if agent id return specific obervation
            pass
        # else return Dict for all agents 
        pass
    
    def is_alive(self, agent_id):
        return self._env._actors[agent_id].is_alive

    def last_actions(self):
        pass

    def last_rewards(self):
        pass

    def last_messages(self):
        pass
    
    def render(self, mode= "human"):
        # TODO complete clear render
        # Add render mode - pyscreen
        print(self._env.render())
