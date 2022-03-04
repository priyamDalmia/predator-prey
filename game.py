import os
import logging
import numpy as np

class Game():
    def __init__(self, size, npred, nprey):
        self.size = size
        self.npred = npred
        self.nprey = nprey
        self.map = Map(self.size)



class Map():
    def __init__(self, size, channels=1):
        self.map = np.zeros(shape=(size, size), dtype=np.int32)
