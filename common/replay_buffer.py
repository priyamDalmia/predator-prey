import random

class ReplayBuffer():
    def __init__(self, memory_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.counter = 0
        # memory 
        self.states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.flaot32)
        self.next_states = np.zeros((self.buffer_size, *self.state_size), dtype=np.float32)
        #self.next_actions = np.zeros((self.
        #self.dones = []
        self.infos = []

    def store_transition(self, **transition):
        index = self.counter % self.buffer_size
        self.states[index] = transition.state
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.next_states[index] = transition.next_state
        self.dones[index] = transition.done

        self.counter += 1

    def sample_batch(self):

        max_index = min(self.counter, self.batch_size)
        batch_ids = np.random.choice(max_index, self.batch_size, replace=False)

        bstates = self.states[batch_ids]
        bactions = self.actiosn[batch_ids]
        brewards = self.rewards[batch_ids]
        bnext_states = self.next_states[batch_ids]
        bdones = self.dones[batch_ids]
        
        return bstates, bactions, brewards, bnext_states, bdones
