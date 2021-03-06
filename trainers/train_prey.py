import os   
import sys
sys.path.append(os.getcwd())
import pdb
import copy
import numpy as np
import pandas as pd
from game.game import Game
from data.helpers import dodict
from data.trainer import Trainer
from data.replay_buffer import ReplayBuffer
# Importing Agents
from data.agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.tor_dqn import DQNAgent
from agents.tor_adv_ac import ACAgent

network_dims=dodict(dict(
    clayers=2,
    cl_dims=[6, 12],
    nlayers=2,
    nl_dims=[256, 256]))
agent_network = dodict(dict(
    network_dims=network_dims))
config = dodict(dict(
        # Environment
        size=10,
        npred=2,
        nprey=1,
        winsize=5,
        nholes=0,
        nobstacles=0,
        _map="random",
        # Training Control
        epochs=50,
        episodes=1,
        train_steps=1,
        update_eps=1,
        max_cycles = 500,
        training=True,
        # Agent Control
        pred_class=RandomAgent,
        prey_class=ACAgent,
        agent_type="actor-critic",
        agent_network=agent_network,
        lr=0.0005, 
        gamma=0.95,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        batch_size=64,
        buffer_size=5000,
        # Models
        load_prey=False, 
        load_predator=False,
        # Log Control
        _name="prey-AC",
        save_replay=True,
        save_checkpoint=True,
        log_freq = 2,
        wandb=False,
        wandb_mode="online",
        wandb_run_name="1v1:10:5:256:0.0005",#"1v1:10:5:256:0.0005",
        project_name="predator-prey-tests",
        msg="Random vs A2C Test: 1v1",
        notes="single-prey-tests",
        log_level=10,
        log_file="logs/prey.log",
        print_console = True,
        # Checkpoint Control 
        ))

class train_prey(Trainer):
    def __init__(self, config, env, **env_specs):
        super(train_prey, self).__init__(config)
        self.config = config
        self.env = env
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        self.logger = self.get_logger()
        # Initialize the agent
        self.agent_ids = env.agent_ids
        self.agents = self.initialize_agents()
        # Fix: Checkpoint states for saving policies  
        self.checkpnt_state = None
        self.steps_avg = 0
        self.rewards_avg = 0
        self.loss_avg = 0
    
    def train(self):
        steps_hist = []
        rewards_hist = []
        loss_hist = []
        _best = 100
        for epoch in range(self.config.epochs):
            loss = [[0, 0]]
            # Run Episodes
            steps, rewards, epsilon = self.run_episodes()
            # Train Agents 
            if self.config.training:
                loss = self.run_training(ep_end=True)
            # Any Agent Specific Update goes here.
            if (epoch%self.config.update_eps):
                for _id in self.agents:
                    self.agent[_id].update_epsilon()
            # Logging results
            steps_hist.append(steps)
            rewards_hist.extend(rewards)
            loss_hist.extend(loss)
            if ((epoch+1)%self.config.log_freq) == 0:
                # Make Checkpoints, Save Replays and Update Logs. 
                self.make_log(epoch, steps_hist, rewards_hist, loss_hist)
                if self.config.save_checkpoint:
                    if _best < self.steps_avg:
                        # Save Agent Network State Dict.
                        self.make_checkpoint(epoch) 
                        _best = self.steps_avg
                        if self.config.save_replay:
                            # Make a Replay File
                            replay_file = f"{self.config._name}-{epoch}-{int(self.steps_avg)}"
                            self.env.record_episode(replay_file)     
        # Save the best model after training
        if self.config.save_checkpoint:
            for _id in self.agent_ids:
                if _id.startswith("prey_"):
                    self.agents[_id].save_model()

    def initialize_agents(self):
        agents = {}
        memory = ReplayBuffer(
                self.config.buffer_size,
                self.config.batch_size, 
                self.input_dims)
        for _id in self.agent_ids:
            if _id.startswith("prey"):
                agent = self.config.prey_class(
                    _id,  
                    self.input_dims,
                    self.output_dims, 
                    self.action_space,
                    memory = memory,
                    load_model = self.config.load_pred,
                    **self.config)
            else:
                agent = self.config.pred_class(
                    _id, 
                    self.input_dims,
                    self.output_dims,
                    self.action_space,
                    memory = None,
                    load_model = self.config.load_prey,
                    **self.config)
                self.log("Agent {_id}, Device {agent.device}")
            assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
            agents[_id] = agent
        return agents

    def run_episodes(self):
        step_hist = []
        reward_hist = []
        epsilon = 0
        for ep in range(self.config.episodes):
            observation = dict(self.env.reset())
            done = False
            steps = 0
            all_rewards = []
            while not done:
                actions = {}
                actions_prob = {}
                state_t = None
                # Get actions for all agents.
                for _id in self.agents:
                    actions[_id], actions_prob[_id] = \
                            self.agents[_id].get_action(observation[_id])
                states_t = copy.deepcopy(observation)
                rewards, next_, done, info = self.env.step(actions)
                for _id in self.agents:
                    self.agents[_id].store_transition(states_t[_id],
                        actions[_id],
                        rewards[_id],
                        next_[_id],
                        done,
                        actions_prob[_id])
                all_rewards.append(list(rewards.values()))
                observation = dict(next_)
                steps+=1
                if steps > self.config.max_cycles:
                    break
            epsilon = self.agents["prey_0"].epsilon
            step_hist.append(steps)
            reward_hist.append(
                    pd.DataFrame(all_rewards).sum(axis=0).to_list())
        return step_hist, reward_hist, epsilon
    
    def run_training(self, ep_end):
        loss_hist = []
        for i in range(self.config.train_steps):
            for _id in self.agents:
                if _id.startswith("prey"):
                    loss = self.agents[_id].train_step()
            loss_hist.append(loss)
        return loss_hist

    def make_log(self, epoch, steps_hist, rewards_hist, loss_hist):
        self.steps_avg = np.mean(steps_hist[-99:])
        self.rewards_avg = pd.DataFrame(rewards_hist, columns = self.agent_ids)\
                        .mean(0).round(2).to_dict()
        self.loss_avg = pd.DataFrame(loss_hist, columns=["total_loss", "delta_loss"])\
                        .mean(0).round(2).to_dict()
        info = dict(
                steps=self.steps_avg,
                rewards=self.rewards_avg,
                loss=self.loss_avg)
        self.update_logs(epoch, info=info)
        if self.config.print_console:
                    print(f"Epochs:{epoch:4} | Steps:{self.steps_avg:4.2f}| Rewards:{self.rewards_avg}")
    
    def make_checkpoint(self, epoch):
        for _id in self.agent_ids:
            if _id.startswith("prey_"):
                c_name = f"{_id}-{epoch}-{self.steps_avg:.0f}"
                self.agents[_id].save_state(c_name)
    
    def save_model(self):
        for _id in self.agent_ids:
            if _id.startswith("prey_"):
                self.agents[_id].save_model()

if __name__=="__main__":
    # Create the Environment object.
    try:
        env = Game(config)
    except:
        print(f"Failed to Initialize the Game Environment!")
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    action_space = env.action_space
    trainer = train_prey(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()

