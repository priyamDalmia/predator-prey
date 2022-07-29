from typing import NewType
import random

class ObservationSpace:
    def __init__(self, space, hi: int = 2, low: int = 0):
        self.space = space
        pass

    def __repr__(self):
        return f"{self.space}"

class ActionSpace:
    def __init__(self, space):
        self.space = space
        pass

    def __repr__(self):
        return f"{self.space}"

    def sample(self):
        return random.randrange(0, self.space[0])
