import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras 
import tensorflow.keras.layers as layers


class Network(keras.Model):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims 

        # Layers definition
        self.conv1 = layers.Conv2D(filters=32,
                kernel_size=(2,2),
                activation='relu',
                kernel_initializer='glorot_uniform',
                padding='valid',
                data_format='channels_first',
                dtype=tf.float32,
                input_shape=self.input_dims)

        self.conv2 = layers.Conv2D(filters=32, 
                kernel_size=(2,2),
                activation='relu',
                kernel_initializer='glorot_uniform',
                padding='valid',
                dtype=tf.float32,
                data_format='channels_first')
        
        self.pool = layers.MaxPool2D(pool_size=(2,2))
        self.flatten = layers.Flatten()

        self.layers1 = layers.Dense(256, activation='relu', dtype=tf.float32)
        self.layers2 = layers.Dense(128, activation='relu', dtype=tf.float32)
        self.outputs_probs = layers.Dense(self.output_dims, activation="softmax")

    def call(self, inputs):
        x = self.pool(self.conv1(inputs))
        x = self.pool(self.conv2(x))
        
        x = self.flatten(x)
        x = self.layers1(x)
        x = self.layers2(x)
        probs = self.outputs_probs(x)

        return probs
    

class Agent():
    def __init__(self, input_dims, output_dims, input_space, load_model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_space = input_space
        self.load_model = load_model
        self.lr = 0.01
        self.epsilon = 0
        self.replay_memory = None
        self.network = None
        

        if load_model:
            pass
        else:
            self.network = Network(self.input_dims, self.output_dims)
            # Compile network here.

        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))

    def get_action(self, observation):
        # add support for exploration.

        observation = tf.convert_to_tensor([observation], dtype=tf.float32)

        probs = self.network(observation)
        action_dist = tfp.distributions.Categorial(probs)
        actions = action_dist.sample()
        return actions

    def train_on_batch(self):
        pass
    
