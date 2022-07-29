import os 
import sys
sys.path.append(os.getcwd())

import random
import numpy as np

from data.game import Environment
from data.config import Config

from game.utils import Actor, Predator, Prey, Wall
from typing import Tuple, Dict

NUM_CHANNELS = 3
PREDATOR_ACTION_SPACE = (4,)
PREY_ACTION_SPACE = (4,)

class SimplePP(Environment):
    def __init__(self, config: Config):
        self._config = config
        self.map_size = config.map_size
        self.observation_space = None
        self.action_space = None
        self.make_base_state()
        self.make_actors()
        self.make_state()
        # build action groups 
        super().__init__(self.state_space, self.actors, self.make_metadata())

    def reset(self):
        self.make_actors()
        self.make_state()
        observations = {}
        for actor_id in self.actors:
            observation[actor_id] = self.make_observation(actor_id)
        return observations

    def step(self, actions: Dict):
        # always pass a dict with _ids and corresponding actions
        observations = {}
        rewards = self.base_rewards.copy()
        dones = self.base_dones.copy()
        for actor_id in self.action_order:
            actor = self.actors[actor_id]
            if not actor.is_alive:
                dones[actor_id] = 1 - actor.is_alive
                observations[actor_id] = self.make_observations(actor_id)
                continue

            new_position = self.actor.make_action(actions[actor_id])
            actor, obj = self.check_collisions()
            if obj:
                if isinstance(obj, Wall):
                    dones[actor_id] = actor.is_alive
                    rewards[actor_id] += 0
                    observations[actor_id] = self.make_observations(actor_id)
                    continue

                if isinstance(obj, Predator):
                    if isinstance(actor, Predator):
                        dones[actor_id] = 1 - actor.is_alive
                        observations[actor_id] = self.make_observations(actor_id)
                        continue
                    # kill prey and update rewards
                    obj.is_alive = False
                    rewards[obj] -= 1
                    dones[str(obj)] = 1 - obj.is_alive
                    rewards[actor_id] += 1
                    dones[actor_id] =  1 - actor.is_alive
                    observations[actor_id] = self.make_observations(actor_id)

                if isinstance(obj, Prey):
                    if isinstance(actor, Predator):
                        # kill prey and update rewards
                        actor.is_alive = False
                        rewards[actor_id] -= 1
                        dones[actor_id] = 1 - actor.is_alive
                        observations[actor_id] = self.make_observations(actor_id)
                        rewards[obj] += 1
                        continue
                    
                    dones[actor_id] = 1 - actor.is_alive
                    observations[actor_id] = self.make_observations(actor_id)
                    continue
            
        actor.position = new_position
        info = []
        return observations, rewards, dones, info

    def make_metadata(self):
        # fill and add more data here
        metadata = {}
        metadata["npred"] = self.npred
        metadata["nprey"] = self.nprey
        return metadata

    def make_observation(self, actor_id: str):
        actor = self.actors[actor_id]
        observation = np.array(None)
        if actor.is_alive:
            if isinstance(actor, Predator):
                breakpoint()
            else:
                breakpoint()
        return observation

    def make_base_state(self):
        # if config.map: then load map else
        # base_channel + padding
        self.pad_width = max(self._config.pred_vision, self._config.prey_vision)
        arr = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        self.base_state = np.pad(arr, pad_width=self.pad_width, constant_values=1)
        self.base_channel = np.pad(arr, pad_width=self.pad_width, constant_values=0)

    def make_actors(self):
        self.npred = self._config.npred
        self.nprey = self._config.nprey
        self.num_actors = self.npred + self.nprey
        self.base_dones = {}
        self.base_rewards = {}
        start_positions = []
        predator_observation_space = (NUM_CHANNELS, self.config.pred_vision, self.config.pred_vision)
        prey_observation_space = (NUM_CHANNELS, self.config.prey_vision, self.config.prey_vision)
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
            self.base_rewards[str(actor)] = 1.0
            self.base_dones[str(actor)] = 1 - actor.is_alive
            self.actors[str(actor)] = actor

    def make_state(self):
        # a numpy array of N channels 
        # Channel #0 : Walls and Obstacles (base state)
        # Channel #2 : Predators
        # Channel #3 : Preys
        # TODO add adjust and write position
        channels = []
        channel_0 = np.copy(self.base_state)
        channels.append(channel_0)
        # for all predators and locations 
        channel_1 = np.copy(self.base_channel)
        for _, actor in self.actors.items():
            if isinstance(actor, Predator):
                pos_x, pos_y = actor.position
                channel_1[pos_x, pos_y] = 1
        channels.append(channel_1)
        # for all preys and locations
        channel_2 = np.copy(self.base_channel)
        for actor in self.actors:
            if isinstance(actor, Prey):
                pos_x, pos_y = actor.position
                channel_2[pos_x, pos_y] = 1
        channels.append(channel_2)
        # stack channels and create the complete state
        self.env_state = np.dstack(channels)

    def check_base_collision(self, pos_x, pos_y) -> bool:
        if self.base_state[pos_x, pos_y] == 1:
            return True
        else:
            return False

    def check_actor_collision(self, position: Tuple) -> Actor:
        # check collisions 
        # and return the position of _id of collided
        channel_idx = None
        if self.env_state[channel_idx, position[0], position[1]]: 
            for actor in self.actors:
                if actor.position == position and actor.is_alive:
                    return actor

