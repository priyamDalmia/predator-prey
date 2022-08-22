import os 
import sys
sys.path.append(os.getcwd())
from enum import Enum
import random
import numpy as np

from data.game import Environment
from data.config import Config

from variations.utils import *
from typing import Dict

NUM_CHANNELS = 3
PREDATOR_CHANNEL = 1
PREY_CHANNEL = 2
PREDATOR_ACTION_SPACE = (4,)
PREY_ACTION_SPACE = (4,)

# ACTIONS (0, UP), (1, DOWN), (2, RIGHT), (3, LEFT)
ACTION_TO_STRING = {
        0: "UP",
        1: "DOWN",
        2: "RIGHT",
        3: "LEFT",
        }

# GRID (0, 0) : UPPER LEFT, (N , N) : LOWER RIGHT.
ACTIONS = {
        0: lambda pos_x, pos_y: (pos_x - 1, pos_y),
        1: lambda pos_x, pos_y: (pos_x + 1, pos_y),
        2: lambda pos_x, pos_y: (pos_x, pos_y + 1),
        3: lambda pos_x, pos_y: (pos_x, pos_y - 1),
        }

class REWARD_MODES(Enum):
    individual = reward_individual
    team = reward_team
    distance = reward_distance

class ACTION_MODES(Enum):
    static = lambda x: x
    random = random.shuffle
    advantage_prey = action_group_prey_first
    advantage_predator = action_group_predator_first

class HEALTH_MODES(Enum):
    standard = 1
    predator = (100, 1)

class TIME_MODES(Enum):
    standard = 0
    time_mode = 500

class SimplePP(Environment):
    def __init__(self, config: Config):
        self._config = config
        self.map_size = config.map_size
        self.make_base_state()
        self.reset()
        # build action groups 
        self.NUM_CHANNELS = NUM_CHANNELS
        self.PREDATOR_CHANNEL = PREDATOR_CHANNEL 
        self.PREY_CHANNEL = PREY_CHANNEL
        
        self.time_mode = False
        self.health_mode = "standard"
        self.action_mode = config.action_mode
        self.reward_mode = config.reward_mode

        super().__init__(self._state_space, self.actors, self.make_metadata())

    def reset(self):
        self._steps = 0
        self.make_actors()
        self.observations = self.make_observations()
        return self.observations

    def step(self, actions: Dict):
        # always pass a dict with _ids and corresponding actions
        observations = {}
        rewards = self.base_rewards.copy()
        dones = self.base_dones.copy()
        action_order = self.action_mode(list(self.actors.keys()))
        for actor_id in action_order:
            actor = self.actors[actor_id]
            action = actions[actor_id] 
            if not actor.is_alive:
                dones[actor_id] = 1 - actor.is_alive
                continue
            new_position = ACTIONS[action](*actor.position)
            obj = self.check_collision(new_position)
            if obj:
                if isinstance(obj, Wall):
                    # if actor hits wall - nothing
                    continue

                if isinstance(obj, Predator):
                    if isinstance(actor, Predator):
                        # predator hits predator - nothing
                        continue
                    # prey hits predator - kill prey and update rewards
                    actor.is_alive = False
                    rewards[actor_id] -= 1
                    dones[actor_id] = 1 - actor.is_alive
                    rewards[str(obj)] += 1
                    continue

                if isinstance(obj, Prey):
                    if isinstance(actor, Predator):
                        # predator hits prey - kill prey and update rewards
                        obj.is_alive = False
                        rewards[str(obj)] -= 1
                        dones[str(obj)] = 1 - actor.is_alive
                        rewards[actor_id] += 1
                        continue
                    # prey hits prey - nothing        
                    continue
            actor.position = new_position
        observations = self.make_observations()
        self.actions = actions
        self.rewards = rewards
        self._dones = dones 
        info = []
        self._steps += 1
        return observations, rewards, dones, info

    def make_metadata(self):
        # fill and add more data here
        metadata = {}
        metadata["npred"] = self.npred
        metadata["nprey"] = self.nprey
        return metadata

    def make_observations(self):
        # update the _env_state
        self.make_state()
        observations = {}
        for actor_id, actor in self.actors.items():
            observation = np.array(None)
            if actor.is_alive:
                pos_x, pos_y = actor.position
                if isinstance(actor, Predator):
                    vision = self._config.pred_vision
                    observation = self._env_state[pos_x-vision:pos_x+vision+1,\
                            pos_y-vision:pos_y+vision+1, :]
                else:
                    vision = self._config.prey_vision
                    observation = self._env_state[pos_x-vision:pos_x+vision+1,\
                            pos_y-vision:pos_y+vision+1, :]
            observations[actor_id] = observation.copy()
        return observations

    def make_base_state(self):
        # if config.map: then load map else
        # base_channel + padding
        self.pad_width = max(self._config.pred_vision, self._config.prey_vision)
        arr = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        self.base_state = np.pad(arr, pad_width=self.pad_width, constant_values=1)
        self.base_channel = np.pad(arr, pad_width=self.pad_width, constant_values=0)
        self._state_space = (NUM_CHANNELS, self.map_size, self.map_size)

    def make_actors(self):
        self.npred = self._config.npred
        self.nprey = self._config.nprey
        self.num_actors = self.npred + self.nprey
        self.base_dones = {}
        self.base_rewards = {}
        start_positions = []
        predator_observation_space = (NUM_CHANNELS, self._config.pred_vision, self._config.pred_vision)
        prey_observation_space = (NUM_CHANNELS, self._config.prey_vision, self._config.prey_vision)
        # if start_position == random
        # generate a list of unique tuples and poplate
        while len(start_positions) != self.num_actors:
            pos_x, pos_y =\
                    (random.randint(0, self.map_size),\
                    random.randint(0, self.map_size))
            # if v clashes with a wall then restart
            if (pos_x, pos_y) not in start_positions and\
                    not self.check_base_collision(pos_x, pos_y):
                start_positions.append((pos_x, pos_y))
        # if start_position == corners 
        # generate a list of unqiue tuples (sampling from corners) and populate 
        self.actors = {}
        self.actor_positions = {}
        # make Predator Actors 
        for idx, position in enumerate(start_positions):
            if idx < self.npred:
                actor = Predator(
                        f"predator_{idx}", 
                        position, 
                        predator_observation_space,
                        PREDATOR_ACTION_SPACE)
            else:
                actor = Prey(
                        f"prey_{idx-self.npred}", 
                        position,
                        prey_observation_space,
                        PREY_ACTION_SPACE)
            self.base_rewards[str(actor)] = 0.0
            self.base_dones[str(actor)] = 1 - actor.is_alive
            self.actors[str(actor)] = actor
        self.rewards = self.base_rewards.copy()
        self._dones = self.base_dones.copy()

    def make_state(self):
        # a numpy array of N channels 
        # Channel #0 : Walls and Obstacles (base state)
        # Channel #2 : Predators
        # Channel #3 : Preys
        channels = []
        channel_0 = np.copy(self.base_state)
        channels.append(channel_0)
        # for all predators and locations 
        channel_1 = np.copy(self.base_channel)
        for _, actor in self.actors.items():
            if isinstance(actor, Predator):
                if actor.is_alive:
                    pos_x, pos_y = actor.position
                    channel_1[pos_x, pos_y] = 1
        channels.append(channel_1)
        # for all preys and locations
        channel_2 = np.copy(self.base_channel)
        for _, actor in self.actors.items():
            if isinstance(actor, Prey):
                if actor.is_alive:
                    pos_x, pos_y = actor.position
                    channel_2[pos_x, pos_y] = 1
        channels.append(channel_2)
        # stack channels and build the complete state
        self._env_state = np.dstack(channels)

    def check_collision(self, position):
        # FIXME limited and naive
        if sum(self._env_state[position[0], position[1], :]): 
            if self.check_base_collision(position[0], position[1]):
                return Wall()
            return self.check_actor_collision(position[0], position[1])
        return None
        
    def check_base_collision(self, pos_x, pos_y) -> bool:
        if self.base_state[pos_x, pos_y] == 1:
            return True
        return False

    def check_actor_collision(self, pos_x, pos_y) -> Actor:
        for _, actor in self.actors.items():
            if actor.position == (pos_x, pos_y) and actor.is_alive:
                return actor
        return False

    def action_to_string(self, action: int):
        return ACTION_TO_STRING[action]
