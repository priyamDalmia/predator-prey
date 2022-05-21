import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras 
import tensorflow.keras.layers as layers

class NetworkConv(keras.Model):
    def __init__(self, input_dims, output_dims):
        super(NetworkConv, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims 

        # Layers definition
        self.conv1 = layers.Conv2D(filters=32,
                kernel_size=(2,2),
                activation='relu',
                data_format='channels_first',
                dtype=tf.float32,
                input_shape=self.input_dims)

        self.conv2 = layers.Conv2D(filters=32, 
                kernel_size=(2,2),
                activation='relu',
                padding="same",
                dtype=tf.float32,
                data_format='channels_first')
        
        self.pool = layers.MaxPool2D(pool_size=(2,2), padding="same")
        self.flatten = layers.Flatten()

        self.layers1 = layers.Dense(256, activation='relu', dtype=tf.float32)
        self.layers2 = layers.Dense(128, activation='relu', dtype=tf.float32)
        self.outputs_probs = layers.Dense(self.output_dims)

    def call(self, inputs):
        x = self.pool(self.conv1(inputs))
        x = self.pool(self.conv2(x))
        
        x = self.flatten(x)
        x = self.layers1(x)
        x = self.layers2(x)
        probs = self.outputs_probs(x)

        return probs

class NetworkLinear(keras.Model):
    def __init__(self, input_dims, output_dims):
        super(NetworkLinear, self).__init__()
        self.input_dims = input_dims
        self.ouptut_dims = output_dims

        self.layer1 = layers.Dense(512, activation='relu', dtype=tf.float32)
        self.layer2 = layers.Dense(512, activation='relu', dtype=tf.float32)
        self.layer3 = layers.Dense(self.ouptut_dims, activation=None)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class DQNAgent():
    def __init__(self, input_dims, output_dims, input_space, load_model, **kwargs):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_space = input_space
        self.load_model = load_model
        self.lr = 0.01
        self.gamma = 0.95
        self.memory = kwargs["memory"]
        self.network = None

        self.epsilon = 0.95
        self.epsilon_dec = 0.90
        self.epsilon_min = 0.1

        self.losses = []
        self.train_step = 0
        
        if load_model:
            pass
        else:
            if isinstance(self.input_dims, int):
                self.network = NetworkLinear(self.input_dims, self.output_dims)
            else:
                self.network = NetworkConv(self.input_dims, self.output_dims)
            # Compile network here.

        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss= "mean_squared_error")

    def get_action(self, observation):
        # add support for exploration.
        if np.random.random() < self.epsilon:
            return np.random.choice(self.input_space)
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        values = self.network(observation)
        action = np.argmax(values)
        return action

    def train_on_batch(self):
        # only train after 500 transitions have been recorded.
        if self.memory.counter < 300:
            return None
        
        # create a index (len = batch_size)
        batch_idx = np.arange(self.memory.batch_size, dtype=np.int32)
        
        # sample from memory (buffer)
        states, actions, rewards, next_states, dones = self.memory.sample_batch()
        
        # define training loop here. 
        values_t0 = self.network(states)
        values_t1 = self.network(next_states)

        target_values = np.copy(values_t0)
        target_values[batch_idx, actions] = rewards + \
                self.gamma * np.max(values_t1, axis=1) * dones
       
        loss = self.network.train_on_batch(states, target_values, return_dict=True)
        # retrive loss and return
        if (self.train_step%50) == 0:
            self.epsilon = (self.epsilon*self.epsilon_dec) if \
                    self.epsilon > self.epsilon_min else self.epsilon_min    
        self.train_step += 1
        self.losses.append(loss)
        return loss 
   
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state = state,
                action = action,
                reward = reward, 
                next_state = next_state,
                done = done)
    
    def load_model(self):
        pass
