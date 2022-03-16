import os 
os.environ["MIN_CPP_LOG_LEVEL"] = '2'

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras 

class Network(keras.Model):
    pass
    

class Agent():
    def __init__(self, input_dims, output_dims, input_space, load_model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_space = input_space
        self.load_model = load_model

    def get_action(self, observation):
        return 0

