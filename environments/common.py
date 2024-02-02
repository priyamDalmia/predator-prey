import warnings
import numpy as np
from typing import Any  
import math 
import random 

# GLOBAL STATE
NUM_CHANNELS = 3
GROUND_CHANNEL = 0
PREDATOR_CHANNEL = 1
PREY_CHANNEL = 2

# AGENTS OBSERVE EXTRA CHANNEL
SELF_CHANNEL = 3

# GRID (0, 0) : UPPER LEFT, (N , N) : LOWER RIGHT.
NUM_ACTIONS = 10
MOVE_ACTION = {
    0: lambda pos_x, pos_y: (pos_x, pos_y),  # STAY
    1: lambda pos_x, pos_y: (pos_x - 1, pos_y),  # UP
    2: lambda pos_x, pos_y: (pos_x + 1, pos_y),  # DOWN
    3: lambda pos_x, pos_y: (pos_x, pos_y + 1),  # RIGHT
    4: lambda pos_x, pos_y: (pos_x, pos_y - 1),  # LEFT
}

ROTATE_ACTION = {
    5: lambda direction: 1,
    6: lambda direction: 2,
    7: lambda direction: 3,
    8: lambda direction: 4,
}

DIRECTION_TO_VECTOR = {
    1: [(-1, -1), (0, 0)],
    2: [(1, 1), (0, 0)],
    3: [(0, 0), (1, 1)],
    4: [(0, 0), (-1, -1)],
}

STR_TO_ACTION = {
    "STAY": 0,
    "UP": 1,
    "DOWN": 2,
    "RIGHT": 3,
    "LEFT": 4,
    "ROTATE_UP": 5,
    "ROTATE_DOWN": 6,
    "ROTATE_RIGHT": 7,
    "ROTATE_LEFT": 8,
    "SHOOT": 9,
}

ACTION_TO_STR = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "RIGHT",
    4: "LEFT",
    5: "ROTATE_UP",
    6: "ROTATE_DOWN",
    7: "ROTATE_RIGHT",
    8: "ROTATE_LEFT",
    9: "SHOOT",
}

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

class FixedSwingAgent:
    def __init__(self, env=None) -> None:
        self.direction = random.choice(["LEFT", "RIGHT"])
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D 
            a = int(math.sqrt(observation.shape[0]/3))
            observation = observation.reshape(a,a,3)
        center = observation.shape[0] // 2
        observation = observation.T
        # if close of the left wall, change direction and move 
        if observation[0, center, :center].sum() > 1:
            self.direction = "RIGHT"
            return STR_TO_ACTION[self.direction]
        elif observation[0, center, center:].sum() > 1:
            self.direction = "LEFT"
            return STR_TO_ACTION[self.direction]
        else:
            if np.random.random() < 0.8:
                return STR_TO_ACTION[self.direction]
            else:
                # if close to the top wall, move down
                if observation[0, :center, center].sum() > 1:
                    return STR_TO_ACTION["DOWN"]
                elif observation[0, center:, center].sum() > 1:
                    return STR_TO_ACTION["UP"]
                else:
                    return STR_TO_ACTION[random.choice(["UP", "DOWN"])]
    
    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None

    def get_initial_state(self):
        return 0

class FollowerAgent:
    """
    If predator in vision, takes a step in its direction 
    else, randomly moves
    """
    def __init__(self, env=None) -> None:
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D 
            a = int(math.sqrt(observation.shape[0]/3))
            observation = observation.reshape(a,a,3)
        center = observation.shape[0] // 2

        if observation[:, :, PREDATOR_CHANNEL].sum() > 1:
            for position in zip(*np.where(observation[:, :, PREDATOR_CHANNEL])):
                if position[0] == center and position[1] == center:
                    continue
                elif position[0] < center:
                    return STR_TO_ACTION["LEFT"]
                elif position[0] > center:
                    return STR_TO_ACTION["RIGHT"]
                elif position[1] < center:
                    return STR_TO_ACTION["UP"]
                elif position[1] > center:
                    return STR_TO_ACTION["DOWN"]
        else:
            return STR_TO_ACTION[random.choice(["UP", "DOWN", "LEFT", "RIGHT"])]
    
    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None
    
    def get_initial_state(self):
        return 0 

class AgressiveAgent:
    """
    If predators in direct line of sight, shoots
    If prey in vision, takes a step in its direction (closet prey)
    else, randomly moves
    """

    def __init__(self, env=None) -> None:
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D
            a = int(math.sqrt(observation.shape[0] / 3))
            observation = observation.reshape(a, a, 3)

        center = list(zip(*np.where(observation[:, :, 3] == 1)))[0]
        if observation[:, :, PREDATOR_CHANNEL].sum() > 1:
            c = observation.shape[0] // 2
            if center[0] > c:
                if (
                    observation[
                        : center[0], center[1], PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return STR_TO_ACTION["SHOOT"]
            elif center[0] < c:
                if (
                    observation[
                        center[0] + 1 :, center[1], PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return STR_TO_ACTION["SHOOT"]
            elif center[1] > c:
                if (
                    observation[
                        center[0], : center[1], PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return STR_TO_ACTION["SHOOT"]
            elif center[1] < c:
                if (
                    observation[
                        center[0], center[1] + 1 :, PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return STR_TO_ACTION["SHOOT"]

        if observation[:, :, PREY_CHANNEL].sum() > 0:
            positions = list(zip(*np.where(observation[:, :, PREY_CHANNEL])))
            distance = lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            positions.sort(key=distance)
            for position in positions:
                if position[0] > center[0]:
                    return STR_TO_ACTION["DOWN"]
                elif position[0] < center[0]:
                    return STR_TO_ACTION["UP"]
                elif position[1] < center[1]:
                    return STR_TO_ACTION["LEFT"]
                elif position[1] > center[1]:
                    return STR_TO_ACTION["RIGHT"]
                break
        return np.random.randint(0, 8)

    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None

    def get_initial_state(self):
        return 0


class ChaserAgent:
    """
    If predators in direct line of sight, shoots
    If prey in vision, takes a step in its direction (closet prey)
    else, randomly moves
    """

    def __init__(self, env=None) -> None:
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D
            a = int(math.sqrt(observation.shape[0] / 3))
            observation = observation.reshape(a, a, 3)

        center = list(zip(*np.where(observation[:, :, 3] == 1)))[0]
        if observation[:, :, PREY_CHANNEL].sum() > 0:
            positions = list(zip(*np.where(observation[:, :, PREY_CHANNEL])))
            distance = lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            positions.sort(key=distance)
            for position in positions:
                if position[0] > center[0]:
                    return STR_TO_ACTION["DOWN"]
                elif position[0] < center[0]:
                    return STR_TO_ACTION["UP"]
                elif position[1] < center[1]:
                    return STR_TO_ACTION["LEFT"]
                elif position[1] > center[1]:
                    return STR_TO_ACTION["RIGHT"]
                break
        return np.random.randint(0, 9)

    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None

    def get_initial_state(self):
        return 0



