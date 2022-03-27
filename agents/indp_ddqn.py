import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_probabiliy as tfp
import tensorflow.keras as keras 
from tensorflow.keras import Layers


class Network():
    def __init__(self, input_dims, output_dims):
        self.input_dims = input_dims
        self.output_dims = output_dims 
        
        pass

class DDQNAgnet():
    def __init__(self, input_dims, output_dims, action_space):
        self.input_dims = input_dims
        self.output_dims = output_dims 
        self.action_space = action_space 
        self.learning_rate = learning_rate
        self.gamma = gamma 
        self.epsilon = epsilon
        
        # Initiliaze Memory
        self.memory = memory

        # Initilaize the network 
        self.network = None
        if load_model:
            pass
        else:
            self.network = Network(self.input_dims, self.output_dims)
        
        # compile model here
        self.network.compile()
        pass
    

    def get_action(self, observation):
        pass

    def train(sefl):
        pass

