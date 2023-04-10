from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, _id):
        self._id = _id
        self.input_space = None
        self.output_space = None
        self.aciton_space = None
        self.trains = False
        self.lr = None
        self.gamma = None
        self.memory = None
        self.epsilon = None
        self.device = None

    @abstractmethod
    def get_action(self, observation):
        """
        Takes an observation as input and returns a action
        
        Args:
           observation: current observation of the agent.

        """
        pass
    
    @abstractmethod
    def store_transition(self):
        pass

    @abstractmethod
    def train_step(self):
        pass
    
    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def save_model(self, filename):
        pass

    @abstractmethod
    def load_model(self, filename):
        pass
    
    def clear_loss(self):
        pass

    def update_eps(self):
        pass
