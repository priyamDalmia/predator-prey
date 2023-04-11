import os 
import numpy as np
from data.config import Config
from data.utils import *
from typing import Dict, Tuple

# TODO action to string for each
# TODO build abstract methods and look for abstract variables 

class Environment:
    def __init__(self, state_space: Tuple, actors: Dict, metadata: Dict):
        self._state_space = state_space        
        self._metadata = metadata
        # build observation and action_spaces 
        self.observation_spaces = {}
        self.action_spaces = {}
        for actor_id, actor in self.actors.items():
            self.observation_spaces[actor_id] =\
                    ObservationSpace(actor.observation_space)
            self.action_spaces[actor_id] =\
                    ActionSpace(actor.action_space)
    @property
    def actor_ids(self):
        return list(self.actors.keys())

    @property
    def state_space(self):
        return self._state_space

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
    
    def is_terminal(self):
        raise NotImplementedError
    
    def metadata(self):
        raise NotImplementedError

    def render(self) -> np.array:
        # simple renderer (simplePP); this global renders assumes that channels are in order
        # [GROUND, PREDATOR, PREY]
        # the render also checks if any prey/predator are out of position and prints a warning message if so.
        # for more variations, check overridden render() function.
        lims = self.pad_width
        render_array = self._env_state[0, self.pad_width:(self.pad_width+self.map_size), self.pad_width:(self.pad_width+self.map_size)].copy()
        render_array = render_array.astype(np.str)
        render_array = np.char.replace(render_array, '0', '.')
        render_array = render_array.astype('U11')
        info = dict(
            game_step = 0,
            dead = [],
            last_actions = []
        )
        info["game_step"] = self._steps
        for actor_id, actor in self.actors.items():
            if not actor.is_alive:
                info["dead"].append(actor_id)
                continue
            else:
                info["last_actions"].append(f"{actor_id}:{actor._last_action}")

            if isinstance(actor, Predator):
                name = f"T{actor_id[-1]}"
                position = actor.position
                if not self._env_state[1, position[0], position[1]]:
                    print(f"Predator {actor_id} is out of place.")
                else:
                    render_array[position[0]-lims, position[1]-lims] = name
            
            if isinstance(actor, Prey):
                name = f"P{str(actor_id[-1])}"
                position = actor.position
                if not self._env_state[2, position[0], position[1]]:
                    print(f"Prey {actor_id} is out of place.")
                else:
                    render_array[position[0]-lims, position[1]-lims] = name
        return render_array, info

class GameHistory:
    def __init__(self):
        pass

class Game:
    def __init__(self, config: Config, env: Environment):
        self.config = config
        self.game_config = config.game_config
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
            self._env.actor_observations(actor_id)
        return self._env.actor_observations()

    def is_terminal(self):
        return self._env.is_terminal()
    
    def current_state(self, agent_id = None):
        if agent_id:
            # if agent id return specific obervation
            self._env.actor_observations()
        # else return Dict for all agents 
        return self._env.actor_observations()
    
    def is_alive(self, agent_id):
        return self._env.actors[agent_id].is_alive

    def last_actions(self):
        raise NotImplementedError

    def last_rewards(self):
        return self._env.rewards

    def last_messages(self):
        raise NotImplementedError
    
    def render(self, mode="human"):
        # TODO complete clear render
        # Add render mode - pyscreen
        if mode == "human":
            render, info = self._env.render()
            if self.game_config.render_info:
                print(info)
            print(render)
            
