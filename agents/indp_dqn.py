import os 
os.environ["MIN_CPP_LOG_LEVEL"] = '2'

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras 
import tensorflow.keras.layers as layers

class Network(keras.Model):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.ouput_dims = output_dims

        # Network Architecture here.
        self.layers1 = layers.Dense()
        self.layers2 = layers.Dense()
        self.outputs = layers.Dense(activation="softmax")

    def call(self, inputs):
        x = self.layers1(inputs)
        x = self.layers2(x)
        probs = self.outputs(x)
    

class Agent():
    def __init__(self, input_dims, output_dims, input_space, load_model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_space = input_space
        self.load_model = load_model

        if load_model:
            pass
        else:
            # Init network here
            pass

    def get_action(self, observation):
        breakpoint()
        return 0

