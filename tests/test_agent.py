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
from agents.tor_n_AC import AACAgent
import pdb

agent_network = dodict(dict(
    clayers=2,
    cl_dims=[12, 12],
    nlayers=2,
    nl_dims=[256, 512],
    rnn_layers=2))

config = dodict(dict(
        # Environment
        size=10,
        npred=3,
        nprey=8,
        winsize=9,
        nholes=0,
        nobstacles=0,
        map_="random",
        reward_mode="distance",
        advantage_mode=False,
        distance_factor=0.5,
        time_mode=True,
        steps_limit=300,
        # Training Control
        epochs=1,
        episodes=1,       # Episodes must be set to 1 for training.
        train_steps=1,    # Train steps must be set to 1 for training.
        update_eps=1,
        max_cycles=600,
        nsteps=30,
        chain=3,
        training=True,
        eval_pred=False,
        eval_prey=False,
        train_type="predator",
        # Agent Control
        class_pred=AACAgent,
        class_prey=AACAgent,
        agent_type="actor-critic",
        agent_network=agent_network,
        lr=0.0005, 
        gamma=0.95,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        batch_size=64,
        buffer_size=1500,
        # Models
        replay_dir="tests/results/",
        checkpoint_dir="tests/policies/",
        load_prey=False, # Give complete Path to the saved policy.#'predator_0-1ac-1random-4799-29', # 'prey_0-random-ac-99-135', 
        load_pred=False, #'prey_0-1random-1ac-4799-390', #'predator_0-ac-random-19-83',
        # Log Control
        _name="15-2rnn-5rand-chain1",
        save_replay=False,
        save_model=False,
        log_freq=2,
        wandb=False,
        wandb_mode="offline",
        entity="rl-multi-predprey",
        project_name="rnn-tests",
        notes="Testing Decentralized Individual Agents",
        log_level=10,
        log_file="tests/test_agent.log",
        print_console = True,))

class test_agent(Trainer):
    def __init__(self, config, env, **env_specs):
        super(test_agent, self).__init__(env, config)
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
            # Run Episodes 
            steps, rewards = self.run_episodes()
            # Train Agents 
            loss = [0 for i in range(len(self.train_ids))]
            if self.config.training:
                loss = self.run_training(ep_end=True)
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
            for _id in self.train_ids:
                    self.agents[_id].save_model()

    def make_checkpoint(self, epoch):
        if self.config.train_type == "predator":
            if self.best_ < self.steps_avg:
                return
        else:
            if self.best_ > self.steps_avg:
                return
        self.best_ = self.steps_avg
        # Code To Make Checkpoints
        for _id in self.train_ids:
            c_name = \
                    f"{_id}-{self.config._name}-{epoch}-{self.steps_avg:.0f}"
            self.agents[_id].save_state(self.config.checkpoint_dir+c_name)
        # Save Game Replay for the last game.
        if self.config.save_replay:
            # Make a replay file.
            replay_dir = self.config.replay_dir
            replay_file = f"{self.config._name}-{epoch}-{int(self.steps_avg)}"
            self.env.record_episode(replay_dir+replay_file)     
                
    def initialize_agents(self):
        agents = {}
        for _id in self.pred_ids:
            memory = None
            if self.config.train_type.startswith("predator"):
                memory = ReplayBuffer(
                    self.config.buffer_size,
                    self.config.batch_size, 
                    self.input_dims)
            try:
                agent = self.config.class_pred(
                    _id,  
                    self.input_dims,
                    self.output_dims, 
                    self.action_space,
                    memory = memory,
                    lr = self.config.lr,
                    gamma = self.config.gamma, 
                    load_model = self.config.load_pred,
                    eval_model = self.config.eval_pred,
                    agent_network = self.config.agent_network)
                assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
                self.log_write(f"Agent Created: {_id} | Policy Loaded:{self.config.load_pred}")
                self.log_model(agent.network)
            except Exception as e:
                self.log_write(f"Agent init Failed: {_id} | Policy Loaded:{self.config.load_pred}")
            agents[_id] = agent
        for _id in self.prey_ids:
            memory = None 
            if self.config.train_type.startswith("prey"):
                memory = ReplayBuffer(
                    self.config.buffer_size,
                    self.config.batch_size, 
                    self.input_dims)
            try:
                agent = self.config.class_prey(
                    _id, 
                    self.input_dims,
                    self.output_dims,
                    self.action_space,
                    memory = memory,
                    lr = self.config.lr,
                    gamma = self.config.gamma,
                    load_model = self.config.load_prey,
                    eval_model = self.config.eval_prey,
                    agent_network = self.config.agent_network)
                assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
                self.log_write(f"Agent Created: {_id} | Policy Loaded:{self.config.load_prey}")
                self.log_model(agent.network)
            except Exception as e:
                self.log_write(f"Agent init Failed: {_id} | Policy Loaded:{self.config.load_prey}")
            agents[_id] = agent
        return agents

    def run_episodes(self):
        step_hist = []
        reward_hist = []
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
                        actions[_id], _ = self.agents[_id].get_action(observation[_id])
                    else:
                        actions[_id] = int(4)
                states_t = copy.deepcopy(observation)
                rewards, next_, done, info = self.env.step(actions)
                for _id in train_agents:
                    self.agents[_id].store_transition(states_t[_id],
                            actions[_id],
                            rewards[_id],
                            done_[_id])
                    if done_[_id]:
                        train_agents.remove(_id)
                all_rewards.append(list(rewards.values()))
                all_dones.append(list(done_.values()))
                observation = dict(next_)
                steps+=1
                if steps > self.config.max_cycles:
                    break
            print(steps)
            step_hist.append(steps)
            done_df = pd.DataFrame(all_dones)
            reward_df = \
                    pd.DataFrame(all_rewards)[-done_df].replace(np.nan, 0.0)
            reward_hist.append(
                    pd.DataFrame(reward_df).sum(axis=0).to_list())
            print(reward_hist)
        return step_hist, reward_hist
    
    def run_training(self, ep_end):
        loss_hist = []
        for i in range(self.config.train_steps):
            for _id in self.train_ids:
                loss = self.agents[_id].train_step()
                loss_hist.append(loss)
        return [loss_hist]

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
    trainer = test_agent(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()
