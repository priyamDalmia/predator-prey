import random
import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, state_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_size = state_size
        self.counter = 0
        # memory 
        self.states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32) 
        #self.next_actions = np.zeros((self.
        #self.dones = []
        self.infos = []

    def store_transition(self, state, action, reward, next_state, done):
        index = self.counter % self.buffer_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        if next_state.size > 0:
            self.next_states[index] = next_state
        else:
            self.next_states[index] = state
        self.dones[index] = 1 - int(done)

        self.counter += 1

    def sample_batch(self):
        max_index = min(self.counter, self.batch_size)
        batch_ids = np.random.choice(max_index, self.batch_size, replace=False)

        bstates = self.states[batch_ids]
        bactions = self.actions[batch_ids]
        brewards = self.rewards[batch_ids]
        bnext_states = self.next_states[batch_ids]
        bdones = self.dones[batch_ids]
        
        return bstates, bactions, brewards, bnext_states, bdones

    def clear_buffer():
        self.counter = 0
        # memory 
        self.states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32) 
        #self.next_actions = np.zeros((self.
        #self.dones = []
        self.infos = []
