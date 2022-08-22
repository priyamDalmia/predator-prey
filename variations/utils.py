from typing import List, Tuple, Optional
import numpy as np
import random

# Action and Observation Spaces classes for the games.
# Similar to gym.env.spaces from the OpenAI gym library.
class ActionSpace:
    def __init__(self, n: int, start: int = 0) -> int:
        self.n = n
        self.start = start

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        # TODO create maksed return
        return np.random.choice(self.start + self.n)

    def contains(self, x: int) -> bool:
        return self.start <= x < self.n

class ObservationSpace:
    def __init__(self, space: Tuple):
        self.space = space

# All Objects and Units(Actor) classes for the games.
class Wall:
    def __init__(self, position = None):
        self.position = position

class Actor:
    def __init__(self, _id, position, observation_space, action_space):
        self._id = _id
        self._position = position
        self._observation_space = observation_space
        self._action_space = action_space
        self._is_alive = True
    
    @property
    def position(self) -> Tuple:
        return self._position

    @position.setter
    def position(self, position: Tuple):
        self._position = position
    
    @property
    def observation_space(self) -> Tuple:
        return self._observation_space
    
    @property
    def action_space(self) -> Tuple:
        return self._action_space

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value: bool):
        self._is_alive = value

    def __repr__(self):
        return self._id

class Predator(Actor):
    def __init__(self, _id, position, observation_space, action_space):
        super().__init__(_id, position, observation_space, action_space)
        # TODO define available actions here
        pass
    
class Prey(Actor):
    def __init__(self, _id, position, observation_space, action_space):
        super().__init__(_id, position, observation_space, action_space)
        pass

class Scout(Predator):
    def __init__(self, _id, position, observation_space, action_space):
        super().__init__(_id, position, observation_space, action_space)
        pass

# Action groups for the games
# determines the order in which actors take actions
def action_group_random(actor_ids: List) -> List:
    return random.shuffle(actor_ids)

def action_group_predator_first(actor_ids: List) -> List:
    return actor_ids

def action_group_prey_first(actor_ids: List) -> List:
    pass

# Reward functions for distributing rewards within the team. 
# determines how the reward is distributed among team memebers.
def reward_individual():
    pass

def reward_team():
    pass

def reward_distance():
    pass


# Health Functions for the games
def health_standard():
    pass
