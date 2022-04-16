from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, _id):
        self._id = _id
        self.input_dims = None
        self.output_dims = None
        self.aciton_space = None
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
    def train_on_batch(self):
        pass
    
    @abstractmethod
    def update_eps(self):
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
