import numpy as np
import os
import sys
import random



class ReplayBuffer():
    def __init__(self, memory_size, batch_size):
        self.mem_size = mem_size
        self.batch_size = batch_size

        # memory 
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_actions = []
        self.dones = []
        self.infos = []

    def store_transition(self, **transition):
        pas

    def sample_batch(self):

        if self.counter < self.batch_size * 10:
            return None

        batch_ids = np.random.randint(self.counter, size=self.batch_size)
        bstates = self.states[batch_ids]
        bactions = self.actiosn[batch_ids]
        brewards = self.rewards[batch_ids]
        bnext_states = self.next_states[batch_ids]
        bdones = self.dones[batch_ids]

