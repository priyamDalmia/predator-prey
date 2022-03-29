import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras 
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

'''
An implementation of the Duelling Deep-Q network for the predator-prey environment.
'''

class Network(keras.Model):
    def __init__(self, input_dims, output_dims):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims 
        
        # Network architecture 
        # Cnn layers + Flatten 
        self.cnn1 = layers.Conv2D(filters=32,
                kernel_size=(2,2),
                padding="same",
                data_format="channels_first",
                kernel_initializer = 'random_normal',
                activation="relu",
                dtype=np.float32,
                input_shape=self.input_dims)
        self.cnn2 = layers.Conv2D(filters=16, 
                kernel_size=(2,2),
                padding="valid",
                data_format="channels_first",
                kernel_initializer = 'random_normal',
                activation="relu",
                dtype=np.float32)
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(512, activation='relu', dtype=np.float32)
        
        # Decomposed layers for state values
        self.layer_v1 = layers.Dense(256, activation='relu', dtype=np.float32)
        self.layer_v2 = layers.Dense(1, activation=None)
        
        # Decomposed layers for action-advantage values
        self.layer_a1 = layers.Dense(256, activation='relu', dtype=np.float32)
        self.layer_a2 = layers.Dense(self.output_dims, activation=None)
        
        self.layer_mean = layers.Lambda(lambda a: a[:,:] - K.mean(a[:,:], keepdims=True), output_shape=(self.output_dims, )) 
        self.layer_add = layers.Add()

    def call(self, inputs):
        '''
        returns: (state_value [batch_size*1] + adjusted_advantage[batch_size, n_actions])
        '''
        x = self.cnn1(inputs) 
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)

        # Decomposed network to produce the state-values and advantages.
        values = self.layer_v1(x)
        values = self.layer_v2(values)        
        advantages = self.layer_a1(x)
        advantages = self.layer_a2(x)
        advantages = self.layer_mean(advantages)
        
        # Adding the decoposed outputs to produce the final vector.
        action_values = self.layer_add([values, advantages])
        return action_values

class DDQNAgent():
    def __init__(self, input_dims, output_dims, action_space, memory, learning_rate=0.05,
            gamma=0.9, epsilon=0.9, batch_idx=64, load_model=False):
        self.input_dims = input_dims
        self.output_dims = output_dims 
        self.action_space = action_space 
        self.learning_rate = learning_rate
        self.gamma = gamma 
        
        self.epsilon = epsilon
        self.epislon_dec = 0.05
        self.epsilon_min = 0.1

        # Initiliaze Memory
        self.batch_idx = batch_idx
        self.memory = memory

        # Learning bookkeeping
        self.learn_step = 0
        self.loss = []

        # Initilaize the network 
        self.network = None
        if load_model:
            pass
        else:
            self.network = Network(self.input_dims, self.output_dims)
        
        # compile model here
        self.network.compile(optimizer=Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error')
    

    def get_action(self, observation):

        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        obs = tf.convert_to_tensor([observation], dtype=tf.float32)

        action_values = self.network(obs)
        action = np.argmax(action_values, axis=1)
        return action[0]

    def train_on_batch(self):

        if self.memory.counter < 100:
            return None 
        
        batch_ids = [i for i in range(self.batch_idx)]
        breakpoint()

        # make modification for two different eval and next networks.
        # get batches of data.
        states, actions, rewards, next_states, dones = self.memory.sample_batch()

        # get the states values and the advantages
        values = self.network(tf.convert_to_tensor(states, dtype=np.float32))
        values_ = self.network(tf.convert_to_tensor(next_states, dtype=np.float32))

        targets = values.numpy()
        values_ = values_.numpy()
        values_max = np.argmax(values_, axis=1)
        
        targets[batch_ids, actions] = rewards[batch_ids] + \
                self.gamma * (values_[batch_ids, values_max]) 
        # calculate the next state values
        loss = self.network.train_on_batch(states, targets)

        self.epislon = self.epsilon - self.epislon_dec if \
                self.epislon > self.epsilon_min else self.epsilon    
        self.learn_step += 1

        return loss

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def load_model(self):
        pass
