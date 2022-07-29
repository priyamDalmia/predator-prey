from typing import NewType, Tuple

class Object:
    def __init__(self, position = None):
        self.position = position

Wall = NewType("Wall", str) 

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
