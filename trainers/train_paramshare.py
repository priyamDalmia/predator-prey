import os   
import sys
sys.path.append(os.getcwd())

import yaml
import copy
import pandas as pd
import numpy as np
from game.game import Game
from data.helpers import dodict
from data.trainer import Trainer
from data.replay_buffer import ReplayBuffer
from data.agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.tor_dqn import DQNAgent
from agents.tor_naac import AACAgent

import pdb

class train_agent(Trainer):
    def __init__(self, config, env, **env_specs):
        super(train_agent, self).__init__(env, config)
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        # Initialize the agent
        self.agents = self.initialize_agents()

    def train(self):
        steps_hist = []
        rewards_hist = []
        loss_hist = []
        # Run training Epochs
        for epoch in range(self.config.epochs):
            steps, rewards, loss = self.run_n_training()
            # Logging results
            steps_hist.extend(steps)
            rewards_hist.extend(rewards)
            loss_hist.extend(loss)
            if ((epoch+1)%self.config.log_freq) == 0:
                # Make Checkpoints, Save Replays and Update Logs. 
                self.make_log(epoch, steps_hist, rewards_hist, loss_hist)
                if self.config.save_model:
                    self.make_checkpoint(epoch)
        # Save the best model after training
        if self.config.save_model:
            _id = self.train_ids[-1]
            self.agents[_id].save_model()
            pass
    
    def make_checkpoint(self, epoch):
        if self.config.train_type == "predator":
            if self.best_ < self.steps_avg:
                return
        else:
            if self.best_ > self.steps_avg:
                return
        self.best_ = self.steps_avg
        # Code To Make Checkpoints
        _id = self.train_ids[0]
        c_name = f"{_id}-{self.config._name}-{epoch}-{self.steps_avg:.0f}"
        self.agents[_id].save_state(self.config.checkpoint_dir+c_name)
        # Save Game Replay for the last game.
        if self.config.save_replay:
            # Make a replay file.
            replay_dir = self.config.replay_dir
            replay_file = f"{self.config._name}-{epoch}-{int(self.steps_avg)}"
            self.env.record_episode(replay_dir+replay_file)     
                
    def initialize_agents(self):
        agents = {}
        try:
            agent = self.config.class_pred(
                    "predator",  
                    self.input_dims,
                    self.output_dims, 
                    self.action_space,
                    memory = None,
                    lr = self.config.lr,
                    gamma = self.config.gamma, 
                    load_model = self.config.load_pred,
                    eval_model = self.config.eval_pred,
                    agent_network = self.config.agent_network)
            assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
            self.log_write(f"Agent Created:predator_parashare | Policy Loaded:{self.config.load_pred}")
            self.log_model(agent.network)
        except: 
            self.log_write(f"Agent init Failed:predator_parashare | Policy Loaded:{self.config.load_pred}")
        
        for _id in self.pred_ids:
            agents[_id] = agent
            memory = None
            if self.config.train_type.startswith("predator"):
                memory = ReplayBuffer(
                    self.config.buffer_size,
                    self.config.batch_size, 
                    self.input_dims)
                agents[_id].memory_n[_id] = memory
        try: 
            agent = self.config.class_prey(
                    "prey", 
                    self.input_dims,
                    self.output_dims,
                    self.action_space,
                    memory = None,
                    lr = self.config.lr,
                    gamma = self.config.gamma,
                    load_model = self.config.load_prey,
                    eval_model = self.config.eval_prey,
                    agent_network = self.config.agent_network)
            assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
            self.log_write(f"Agent Created:prey_parashare | Policy Loaded: {self.config.load_prey}")
            self.log_model(agent.network)
        except: 
            self.log_write(f"Agent init Failed:prey_parashare | Policy Loaded:{self.config.load_prey}")
        
        for _id in self.prey_ids:
            agents[_id] = agent
            memory= None
            if self.config.train_type.startswith("prey"):
                memory = ReplayBuffer(
                        self.config.buffer_size,
                        self.config.batch_size, 
                        self.input_dims)
                agents[_id].memory_n[_id] = memory
        return agents
    
    def run_n_training(self):
        step_hist = []
        reward_hist = []
        # Add loss 
        loss_hist = [0 for i in range(len(self.train_ids))]
        for ep in range(self.config.episodes):
            observation, done_ = self.env.reset()
            done = False
            steps = 0
            train_agents = list(self.train_ids)
            all_rewards = []
            all_dones = []
            all_dones.append(list(done_.values()))
            while not done:
                actions = {}
                actions_prob = {}
                state_t = None
                # Get actions for all agents.
                for _id in self.agents:
                    if not done_[_id]:
                        actions[_id], actions_prob[_id] = \
                                self.agents[_id].get_action(observation[_id])
                    else:
                        actions[_id] = int(4)
                        actions_prob[_id] = 0
                states_t = copy.deepcopy(observation)
                rewards, next_, done, info = self.env.step(actions)
                for _id in train_agents:
                    try:
                        self.agents[_id].store_transition(_id, states_t[_id],
                            actions[_id],
                            rewards[_id],
                            next_[_id],
                            done_[_id],
                            actions_prob[_id])
                    except Exception as e:
                        print(e)
                        breakpoint()
                    if done_[_id]:
                        train_agents.remove(_id)
                all_rewards.append(list(rewards.values()))
                all_dones.append(list(done_.values()))
                observation = dict(next_)
                steps+=1
                if steps > self.config.max_cycles:
                    break
                if (steps+1)%self.config.nsteps == 0:
                    # Run training here!
                    losses = self.run_training(ep_end=False)
                    loss_hist = [a+b for a, b in zip(loss_hist, losses)]
            step_hist.append(steps)
            done_df = pd.DataFrame(all_dones)
            reward_df = \
                    pd.DataFrame(all_rewards)[-done_df].replace(np.nan, 0.0)
            reward_hist.append(
                    pd.DataFrame(reward_df).sum(axis=0).to_list())
            losses = self.run_training(ep_end=True)
            loss_hist = [a+b for a, b in zip(loss_hist, losses)]
        return step_hist, reward_hist, [loss_hist]
    
    def run_training(self, ep_end):
        loss_hist = []
        for i in range(self.config.train_steps):
            for _id in self.train_ids:
                loss = self.agents[_id].train_step(_id)
                loss_hist.append(loss)
        return loss_hist

if __name__=="__main__":
    # Parse and Load Custom Config Files here.
    # Create the Environment object.
    try:
        env = Game(config)
    except Exception as e:
        print(e)
        print(f"Failed to Initialize the Game Environment!")
        sys.exit()
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    action_space = env.action_space
    trainer = train_agent(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()
