from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, _id):
        self._id = _id
    
    @abstractmethod
    def get_action(self, observation):
        """
        Takes an observation as input and returns a action
        
        Args:
           observation: current observation of the agent.

        """

        pass

    @abstractmethod
    def train_on_batch(self):
        pass

    @abstractmethod
    def save_model(self, filename):
        pass

    @abstractmethod
    def load_model(self, filename):
        pass
