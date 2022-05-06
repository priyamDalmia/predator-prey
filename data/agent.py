from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, _id):
        self._id = _id
        self.input_dims = None
        self.output_dims = None
        self.action_space = None
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
    def save_state(self, *args):
        pass

    @abstractmethod
    def save_model(self, filename):
        pass

    @abstractmethod
    def load_model(self, filename):
        pass
    
    def discount_rewards():
        new_rewards = []
        _sum = 0
        rewards = np.flip(rewards)
        for i in range(len(rewards)):
            r = rewards[i]
            _sum *= self.gamma
            _sum += r
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        return new_rewards

    def clear_loss(self):
        pass

    def update_eps(self):
        pass
