import warnings 
from typing import Any  

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self._position = None 
        self._is_alive = False 

    def __call__(self, position: tuple) -> Any:
        self._position = position 
        self._is_alive = True
    
    @property
    def is_alive(self):
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value):
        self._is_alive = value
    
    @property
    def position(self):
        if not self.is_alive:
            warnings.warn(f"Requested position for dead agent, {self.agent_id}.")
        return self._position
    
    def move(self, new_position):
        if self.is_alive:
            self._position = new_position
        else:
            warnings.warn(f"Agent {self.agent_id} is dead, cannot move.")
            self._position = None

