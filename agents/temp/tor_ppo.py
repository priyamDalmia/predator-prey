import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from data.agent import BaseAgent

learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20


class Network(nn.Module):
    def __init__(self, 
            input_dims, 
            output_dims, 
            network_dims,
            lr, 
            **kwargs):
        super().__init__()
        # DEFINE MODEL ARCHITECTURE HERE
        #raise NotImplementedError
    
    def forward(self, inputs):
        # THIS FUNCTIONS RETURNS TH ACTIONS PROBABIBLITES 
        breakpoint()
        raise NotImplementedError

    def get_value(self, inputs):
        # THIS FUNCTION WILL RETURN THE VALUE OF A STATE
        # CHECK function v in PPO original file
        pass

class PPOAgent(BaseAgent):
    def __init__(
            self,
            _id,
            input_dims, 
            output_dims, 
            agent_network = {}, 
            **kwargs):
        super().__init__(_id)
        self.input_dims = input_dims
        self.output_dims = output_dims 
        self.agent_network = agent_network
        self.episode_data = []
        ## BUILD NETWORK HERE!
        self.network_dims = agent_network.network_dims
        self.network = Network(
                input_dims, 
                output_dims, 
                self.network_dims,
                lr = 0.01)
    
    def get_action(self, observation):
        # THIS IS THE OBESEVATION AT ANY TIME STEP
        # PASS THE OBSERVATION THROUGH A NEURAL NETWROK 
        action_prob = self.network(observation)
        # AND YOU CONVERT AND RETURN AN ACTION
        action = np.argmax(action_prob)
        return action
    
    def store_transition(self, transition):
        # BASICALLY STORING STATE, ACTION, REWARD, NEXT_STATE
        self.episode_data.append(transition)

    def make_batch(self):
        breakpoint()
        state_list, action_list, reward_list, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        # ASSUMING THAT episode_data contians tuples of transitions 

        for transition in self.episode_data:
            state, action, reward, done, next_state = transition
            action_prob = self.network(state)
            
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            next_state_list.append(next_state)
            action_prob_list.append(action_prob)
        
        # CONVERT TO TENSORS 
        
        states_tensor = torch.tensor(state_list, dtype=torch.float)
        # FOR ALL ATION, REWARDS .. .so no
        return states_tensor, actions_tensor, rewards_tensor, dones_tensor, next_states_tensor, action_probs_tensor
        
    def train_step(self):
        # GETS A LIST OF [STATES], [ACTIONS], [REWARDS], []
        # state, action, reward, 
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        # action_probs = self.network(states)
        breakpoint()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            # this can have significant improvement (efficiency, stability) on performance
            if not np.isnan(advantage.std()):
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5) 
            
            pi = self.pi(s, softmax_dim=-1)
            dist_entropy = Categorical(pi).entropy()
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach()) - 0.01*dist_entropy 

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.episode_data = []

    def load_model(self):
        pass

    def save_model(self):
        pass

    def save_state(self):
        pass

