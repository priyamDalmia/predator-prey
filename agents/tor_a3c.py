import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from data.agent import BaseAgent
import time 
torch.set_default_dtype(torch.float64)

class CriticNet(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(CriticNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.value_layer = nn.LazyLinear(1)
        
    def forward(self, x):
        out = F.relu( self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.value_layer(out)
        return out
    
class ActorNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ActorNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.convout = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.layer_1 = nn.LazyLinear(128)
        self.layer_2 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out = self.convout(out) 
        out = self.flatten(out)
        out = F.relu(self.layer_1(out))
        out = self.layer_2(out)
        out = F.softmax(out, dim=1)
        return out

class A3CAgent(BaseAgent):
    def __init__(self,
                 current_id,
                 input_space,
                 output_space,
                 action_space, 
                 memory,
                 **kwargs):
        super(A3CAgent, self).__init__(current_id)
        self.input_space = input_space
        self.output_space = output_space
        self.action_space = action_space
        
        # use config to dynamically assign 
        self.trains = True
        self.device = "cpu"
        if memory == "self":
            self.clear_memory()
        else:
            self.memory = memory
        self.max_steps = 500
        self.gamma = 0.8
        
        self.actor_net = ActorNet(self.input_space.shape[0], self.action_space.n)
        self.critic_net = CriticNet(self.input_space.shape[0])
        
        self.critic_optimizer = optim.Adam(
                self.critic_net.parameters(),
                lr = 0.0001)
        self.actor_optimizer = optim.Adam( 
                self.actor_net.parameters(),
                lr = 0.0001)

    def get_action(self, observation):
        x = torch.as_tensor(observation, dtype=torch.float64, device=self.device).unsqueeze(0)
        probs = self.actor_net(x)
        action_dist = dist.Categorical(probs)
        return action_dist.sample().item()

    def train_step(self):
        start_time = time.time()
        loss = dict()
        num_steps = len(self.mem_rewards)
        sum_rewards = sum(self.mem_rewards)
        rewards = self.discount_rewards(self.mem_rewards)
        states_t = torch.as_tensor(np.array(self.mem_states),
                                device=self.device, 
                                dtype=torch.float64)             
        actions_t = torch.as_tensor(self.mem_actions).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, 
                                    device=self.device,
                                    dtype=torch.float64).unsqueeze(-1)
        values = self.critic_net(states_t)
        values_next = values.clone().detach()
        values_next = torch.roll(values_next, shifts=-1, dims=0)
        values_next[-1] = 0
        values_target = rewards_t + (values_next * self.gamma)
        critic_loss = torch.nn.functional.mse_loss(values, values_target)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        advantage = (values_target - values).detach()
        probs = self.actor_net(states_t)
        action_probs = torch.gather(probs, 1, actions_t)
        action_log_probs = torch.log(action_probs)
        actor_loss = -1 * action_log_probs * advantage
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor_net.parameters(), 1.0)
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.clear_memory()
        end_time = round(time.time() - start_time, 2)
        return dict(
            actor_loss = actor_loss.item(),
            critic_loss = critic_loss.item(),
            reward = sum_rewards,
            num_steps = num_steps,
        )

        self.clear_memory()
        return loss, dict(
            sum_rewards = sum_rewards
        )
    
    def store_transition(self, transition):
        # transition format -> (s, a, r, d)
        self.mem_states.append(transition[0])
        self.mem_actions.append(transition[1])
        self.mem_rewards.append(transition[2])
        self.mem_dones.append(transition[3])
        self.mem_steps += 1

    def discount_rewards(self, rewards):
        # TODO use max steps to make rewards negative 
        new_rewards = []
        _sum = 0
        for i in range(len(rewards)):
            r = rewards[-1*(i+1)]
            _sum *= self.gamma
            _sum += r
            _sum = round(_sum, 4)
            new_rewards.append(_sum)
        new_rewards = [i for i in reversed(new_rewards)]
        return new_rewards
    

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

    def update_eps(self):
        pass

    def save_state(self, filename):
        pass

    def clear_memory(self):
        self.mem_states = []
        self.mem_actions = []
        self.mem_rewards = []
        self.mem_dones = []
        self.mem_steps = 0
