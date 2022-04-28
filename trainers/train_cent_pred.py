import os   
import sys
sys.path.append(os.getcwd())
import pdb
import copy
import numpy as np
import pandas as pd
import torch
from game.game import Game
from data.helpers import dodict
from data.trainer import Trainer
from data.replay_buffer import ReplayBuffer, Critic_Buffer
# Importing Agents
from data.agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.tor_adv_ac import ACAgent
from agents.tor_cent_ac import CACAgent, NetworkCritic

actor_network = dodict(dict(
    clayers=2,
    cl_dims=[6, 12],
    nlayers=2,
    nl_dims=[256, 256]))

critic_network = dodict(dict(
    clayers=2,
    cl_dims=[6, 12],
    nlayers=2,
    nl_dims=[256, 256]))

config = dodict(dict(
        # Environment
        size=10,
        npred=2,
        nprey=2,
        winsize=5,
        nholes=0,
        nobstacles=0,
        _map="random",
        # Training Control
        epochs=10000,
        episodes=1,       # Episodes must be set to 1 for training.
        train_steps=1,
        update_eps=1,
        max_cycles = 500,
        training=True,
        # Agent Control
        pred_class=CACAgent,
        prey_class=RandomAgent,
        agent_type="cent-AC",
        lr=0.0005, 
        gamma=0.95,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        batch_size=64,
        buffer_size=1500,
        # Models
        load_prey=False, 
        load_predator=False,
        # Log Control
        _name="CAC-rand",
        save_replay=True,
        save_checkpoint=True,
        log_freq = 200,
        wandb=True,
        wandb_mode="online",
        entity="rl-multi-predprey",
        project_name="predator-prey-baselines",
        msg="CAC vs Random Test: 2v2",
        notes="Testing Centralized Training",
        log_level=10,
        log_file="logs/random.log",
        print_console = True,
        # Checkpoint Control 
        ))

class train_pred(Trainer):
    def __init__(self, config, env, **env_specs):
        super(train_pred, self).__init__(config)
        self.config = config
        self.env = env
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        # Initialize the agent
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent_ids = env.agent_ids
        self.agents, self.critic = self.initialize_agents()
        # Fix: Checkpoint states for saving policies  
        self.checkpnt_state = None
        self.steps_avg = 0
        self.rewards_avg = 0
        self.loss_avg = 0

    def train(self):
        steps_hist = []
        rewards_hist = []
        loss_hist = []
        _best = 300
        for epoch in range(self.config.epochs):
            loss = [[0, 0, 0]]
            # Run Episodes
            steps, rewards, epsilon = self.run_episodes()
            # Train Agents 
            if self.config.training:
                loss = self.run_training(ep_end=True)
            # Any Agent Specific Update goes here.
            #if (epoch%self.config.update_eps):
            #    for _id in self.agents:
            #        self.agent[_id].update_epsilon()
            # Logging results
            steps_hist.append(steps)
            rewards_hist.extend(rewards)
            loss_hist.extend(loss)
            if ((epoch+1)%self.config.log_freq) == 0:
                # Make Checkpoints, Save Replays and Update Logs. 
                self.make_log(epoch, steps_hist, rewards_hist, loss_hist)
                if self.config.save_checkpoint:
                    if _best > self.steps_avg:
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
                if _id.startswith("predator_"):
                    self.agents[_id].save_model()

    def initialize_agents(self):
        agents = {}
        # Initialize a Critic Network here!
        # Fix Critic Dims
        critic_input_dims = ((self.config.npred*self.input_dims[0]), \
                self.input_dims[1], self.input_dims[2])
        critic_memory = Critic_Buffer(self.config.buffer_size, self.config.batch_size,
                critic_input_dims)
        critic = NetworkCritic(critic_input_dims, output_dims, memory=critic_memory, 
                network_dims=critic_network, **self.config)
        critic = critic.to(self.device)
        # Move Critic to DEVICE here!
        for _id in self.agent_ids:
            if _id.startswith("predator"):
                memory = ReplayBuffer(
                    self.config.buffer_size,
                    self.config.batch_size, 
                    self.input_dims)
                agent = self.config.pred_class(
                    _id,  
                    self.input_dims,
                    self.output_dims, 
                    self.action_space,
                    memory = memory,
                    load_model = self.config.load_pred,
                    actor_network = actor_network,
                    critic = critic,
                    **self.config)
            else:
                agent = self.config.prey_class(
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
        return agents, critic

    def run_episodes(self):
        step_hist = []
        reward_hist = []
        epsilon = 0
        for ep in range(self.config.episodes):
            observation, done_ = self.env.reset()
            done = False
            steps = 0
            all_agents = list(self.agents.keys())
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
                states_t = copy.deepcopy(observation)
                rewards, next_, done, info = self.env.step(actions)
                for _id in all_agents:
                    self.agents[_id].store_transition(states_t[_id],
                        actions[_id],
                        rewards[_id],
                        next_[_id],
                        done_[_id],
                        actions_prob[_id])
                    if done_[_id]:
                        all_agents.remove(_id)
                # Store Critics Observation Here.
                self.critic.store_transition(states_t, rewards)
                all_rewards.append(list(rewards.values()))
                all_dones.append(list(done_.values()))
                observation = dict(next_)
                steps+=1
                if steps > self.config.max_cycles:
                    break
            step_hist.append(steps)
            done_df = pd.DataFrame(all_dones)
            reward_df = \
                    pd.DataFrame(all_rewards)[-done_df].replace(np.nan, 0.0)
            reward_hist.append(
                    pd.DataFrame(reward_df).sum(axis=0).to_list())
        return step_hist, reward_hist, epsilon
    
    def run_training(self, ep_end):
        """run_training.
        Runs a training loop. Trains the Critic on the combined rewards, and returns the 
        state_values. These state_values are used to compute the advantage using which the 
        agent actor policies are then trained.
        Args:
            ep_end:
        """
        loss_hist = []
        for i in range(self.config.train_steps):
            # Train Critic And Recieve state_valeus and critic_loss
            state_values, critic_loss = self.critic.train_step()
            loss_hist.append(critic_loss)
            for _id in self.agents:
                if _id.startswith("predator"):
                    loss = self.agents[_id].train_step(state_values)
                    loss_hist.append(loss)
        return [loss_hist]

    def make_log(self, epoch, steps_hist, rewards_hist, loss_hist):
        self.steps_avg = np.mean(steps_hist[-99:])
        self.rewards_avg = pd.DataFrame(rewards_hist, columns = self.agent_ids)\
                        .mean(0).round(2).to_dict()
        columns = ["critic_loss", "pred1_loss", "pred2_loss"]
        if self.config.npred > 2:
            breakpoint()
            # Fix the Column names before Procedding.
        self.loss_avg = pd.DataFrame(loss_hist, columns=columns)\
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
            if _id.startswith("predator_"):
                c_name = f"{self.config._name}-{_id}-{epoch}-{self.steps_avg:.0f}"
                self.agents[_id].save_state(c_name)
    
    def save_model(self):
        for _id in self.agent_ids:
            if _id.startswith("predator_"):
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
    trainer = train_pred(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()

