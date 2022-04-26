import random
import numpy as np
import pdb
import gc

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
        self.probs = []
        #self.next_actions = np.zeros((self.
        #self.dones = []
        self.infos = []

    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        index = self.counter % self.buffer_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        if next_state.size > 0:
            self.next_states[index] = next_state
        else:
            self.next_states[index] = state
        self.dones[index] = 1 - int(done)
        if "probs" in kwargs:
            self.probs.append(kwargs["probs"])
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
    
    def sample_transition(self):
        actions= self.actions[:self.counter]
        states = self.states[:self.counter]
        rewards = self.rewards[:self.counter]
        next_ = self.next_states[:self.counter]
        dones = self.dones[:self.counter]
        action_probs = self.probs
        self.clear_buffer()
        return states, actions, rewards, next_, dones, action_probs

    def clear_buffer(self):
        self.counter = 0
        # memory 
        self.states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32) 
        self.probs = []
        #self.next_actions = np.zeros((self.
        #self.dones = []
        self.infos = []

class Critic_Buffer():
    def __init__(self, buffer_size, batch_size, state_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_size = state_size
        self.counter = 0
        # memory 
        self.states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
    
    def clear_buffer(self):
        self.counter = 0
        # memory 
        self.states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
    
    def sample_transition(self):
        states = self.states[:self.counter]
        rewards = self.rewards[:self.counter]
        self.clear_buffer()
        return states, rewards

    def store_transition(self, state, reward):
        index = self.counter % self.buffer_size
        self.states[index] = state
        self.rewards[index] = reward
        self.counter += 1


