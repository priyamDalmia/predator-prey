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
from agents.tor_adv_ac import AACAgent
from agents.tor_coma import COMAAgent, CentCritic

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
        nprey=3,
        winsize=7,
        nholes=0,
        nobstacles=0,
        _map="random",
        # Training control,
        epochs=2000,
        episodes=1,       # Episodes must be set to 1 for training.
        train_steps=1,
        update_eps=1,
        max_cycles=1000,
        training=True,
        train_type="predator",
        update_critic=20,
        # Agent Control
        class_pred=COMAAgent,
        class_prey=RandomAgent,
        agent_type="coma-ac",
        agent_network=actor_network,
        critic_network=critic_network,
        lr=0.0005, 
        gamma=0.95,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        batch_size=64,
        buffer_size=1500,
        # Models
        replay_dir="replays/",
        checkpoint_dir="trained-policies/multi/",
        load_prey=False, 
        load_predator=False,
        # Log Control
        _name="20-t-2coma-5rand",
        save_replay=True,
        save_model=False,
        log_freq=20,
        wandb=True,
        wandb_mode="online",
        entity="rl-multi-predprey",
        project_name="cent-tests",
        notes="2COMA vs 5Rand Cent Pred Test",
        log_level=10,
        log_file="logs/centpred.log",
        print_console=True,
        ))

class train_coma(Trainer):
    def __init__(self, config, env, **env_specs):
        super(train_coma, self).__init__(env, config)
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        # Initialize the agent
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agents, self.critic = self.initialize_agents()

    def train(self):
        steps_hist = []
        rewards_hist = []
        loss_hist = []
        self.best_ = self.steps_avg
        for epoch in range(self.config.epochs):
            # Run Episodes
            steps, rewards = self.run_episodes()
            # Train Agents 
            loss = [0 for i in range(len(self.train_ids)+1)]
            if self.config.training:
                loss = self.run_training(ep_end=True)
            # Update Critic Target netwrok here.
            if ((epoch+1)%self.config.update_critic) == 0:
                self.critic.update_target_critic()
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
        if self.best_ < self.steps_avg:
            return 
        self.best_ = self.steps_avg
        # Make Checkpoints 
        for _id in self.train_ids:
            c_name = \
                    f"{_id}-{self.config._name}-{epoch}-{self.steps_avg:.0f}"
            self.agents[_id].save_state(self.config.checkpoint_dir+c_name)
        if self.config.save_replay:
            # Make a Replay File
            replay_dir = self.config.replay_dir
            replay_file = f"{self.config._name}-{epoch}-{int(self.steps_avg)}"
            self.env.record_episode(repaly_dir+replay_file)  

    def initialize_agents(self):
        agents = {}
        # Initialize a Critic Network here!
        # Fix Critic Dims
        critic_observation_dims = ((self.config.npred*self.input_dims[0]), \
                self.input_dims[1], self.input_dims[2])
        
        critic = CentCritic(
                "critic_0",
                critic_observation_dims, 
                output_dims,
                self.action_space,
                self.pred_ids,
                lr=self.config.lr, 
                gamma=self.config.gamma,
                load_model=self.config.load_model,
                eval_model=self.config.eval_model,
                critic_network=critic_network)
        
        for _id in self.pred_ids:
            memory = ReplayBuffer(
                self.config.buffer_size,
                self.config.batch_size, 
                self.input_dims)
            agent = COMAAgent(
                _id,  
                self.input_dims,
                self.output_dims,
                self.action_space, 
                lr = self.config.lr,
                gamma = self.config.gamma,
                load_model = self.config.load_pred,
                eval_model = self.config.eval_model,
                agent_network = self.config.agent_network)
            assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
            agents[_id] = agent
        for _id in self.prey_ids:
            agent = self.config.class_prey(
                _id, 
                self.input_dims,
                self.output_dims,
                self.action_space)
            assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
            agents[_id] = agent
        return agents, critic

    def run_episodes(self):
        step_hist = []
        reward_hist = []
        for ep in range(self.config.episodes):
            observation, done_ = self.env.reset()
            done = False
            steps = 0
            train_agents = self.train_ids
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
                for _id in train_agents:
                    # Save the Combined Rewards for all the agents instead.
                    if not done_[_id]:
                        self.agents[_id].store_transition(states_t[_id],
                            actions[_id],
                            rewards[_id],
                            next_[_id],
                            done_[_id],
                            actions_prob[_id])
                    else:
                        train_agents.remove(_id)
                # Store Critics Observation Here.
                self.critic.store_transition(states_t, actions, rewards)
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
        return step_hist, reward_hist
    
    def run_training(self, ep_end):
        """run_training.
        Runs a training loop. Trains the Critic on the combined rewards, and returns the 
        Q_values dictionary which contains the agent specific q_values. 
        These Q_values are used to compute the advantage using which the 
        agent actor policies are then trained.
        Args:
            ep_end:
        """
        loss_hist = []
        for i in range(self.config.train_steps):
            # Train Critic And Recieve state_valeus and critic_loss
            critic_loss, Q_values = self.critic.train_step()
            for _id in self.train_ids:
                loss = self.agents[_id].train_step(Q_values[_id])
                loss_hist.append(loss)
            loss_hist.append(critic_loss)
        return [loss_hist]

    def make_log(self, epoch, steps_hist, rewards_hist, loss_hist):
        self.steps_avg = np.mean(steps_hist[-99:])
        self.rewards_avg = pd.DataFrame(rewards_hist, columns = self.agent_ids)\
                        .mean(0).round(2).to_dict()
        columns = [f"pred_{i}_loss" for i in range(self.config.npred)] + ["critic_loss"]
        self.loss_avg = pd.DataFrame(loss_hist, columns=columns).mean(0).round(2).to_dict()
        info = dict(
                steps=self.steps_avg,
                rewards=self.rewards_avg,
                loss=self.loss_avg)
        self.update_logs(epoch, info=info)
        if self.config.print_console:
            print(f"Epochs:{epoch:4} | Steps:{self.steps_avg:4.2f}| Rewards:{self.rewards_avg}")
    
if __name__=="__main__":
    # Create the Environment object.
    try:
        env = Game(config)
    except:
        print(f"Failed to Initialize the Game Environment!")
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    action_space = env.action_space
    trainer = train_coma(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()

