import os
import sys
import copy
import numpy as np
import pandas as pd
from game.game import Game
from data.helpers import dodict
from data.trainer import Trainer
from data.agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.tor_adv_ac import AACAgent
import pdb

# Must have list of Agents classe and their policy paths
prey_class = [AACAgent]
prey_policies = ['prey_0-t-1rand-1ac-19-185']
pred_class = [RandomAgent]
pred_policies = ['random']


config = dodict(dict(
    # Environment
    size=10,
    npred=1,
    nprey=1,
    winsize=5,
    nholes=0,
    nobstacles=0,
    map_="random",
    # Evaluation Control
    runs=5,
    episodes=50,
    max_cycles=500,
    save_replay=False,
    # Agent Control
    print_console=True,
    _name="",
    notes="",
    ))

class Evaluate():
    def __init__(self, config, env, **env_specs):
        self.env = env
        self.config = config
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        # Initialize Agents (Load Agents)
        self.agent_ids = env.agent_ids
        self.agents = self.initialize_agents()
        # Bookeeping
        self.steps_avg = 0
        self.rewards_avg = 0
        self.loss_avg = 0
    
    def evaluate(self, mode="evaluation"):
        for r in range(self.config.runs):
            steps, rewards = self.run_episodes()
            # Make Log                 
            self.make_log(r, steps, rewards)
            # Save the Last episodes of each run!
            if self.config.save_replay:
                replay_file = f"eval-{self.config._name}"
                self.env.record_episode(replay_file)
                        
    def run_episodes(self):
        steps_hist = []
        reward_hist = []
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
                for _id in self.agents:
                    if not done_[_id]:
                        actions[_id], _ =\
                                self.agents[_id].get_action(observation[_id])
                    else:
                        actions[_id] = int(4)
                rewards, next_, done, info = self.env.step(actions)
                all_rewards.append(list(rewards.values()))
                all_dones.append(list(done_.values()))
                observation = dict(next_)
                steps += 1
                if steps > self.config.max_cycles:
                    break
            steps_hist.append(steps)
            done_df  = pd.DataFrame(all_dones)
            reward_df = \
                    pd.DataFrame(all_rewards)[-done_df].replace(np.nan, 0.0)
            reward_hist.append(
                    pd.DataFrame(reward_df).sum(axis=0).to_list())
        return steps_hist, reward_hist
    
    def run_inf_qvals(self):
        # Create A loop
        # Load a Map 
        # Set Prey Action to 4:do_nothing
        # 
        pass

    def make_log(self, r, steps, rewards):
        # Print to console
        if self.config.print_console:
            steps_avg = np.mean(steps[-self.config.episodes:])
            breakpoint()
            rewards_avg = pd.DataFrame(rewards[-self.config.episodes:], columns=self.agent_ids)\
                    .mean(0).round(1).to_dict()
            print(f"Run:{r:4} | Steps:{steps_avg} | Rewards:{rewards_avg}")

    def initialize_agents(self):
        agents = {} 
        # load predator and prey policies 
        for n, _id in enumerate(self.agent_ids):
            if _id.startswith("predator"):
                if len(pred_class) == 1:
                    agent_class = pred_class[0]
                    agent_policy = pred_policies[0]
                else:
                    assert len(pred_class) == self.config.npred, "Error loading agents!, fix policy names"
                    agent_class = pred_class[n]
                    agent_policy = pred_policies[n]
                agent = agent_class(_id, 
                            self.input_dims, 
                            self.output_dims,
                            self.action_space,
                            memory=None,
                            load_model=agent_policy,
                            **self.config)
            else:
                if len(prey_class) == 1:
                    agent_class = prey_class[0]
                    agent_policy = prey_policies[0]
                else:
                    assert len(pred_class) == self.config.nprey, "Error loading agents!, fix policy names."
                    agent_class = prey_class[n]
                    agent_policy = prey_policies[n]
                agent = agent_class(_id, 
                            self.input_dims, 
                            self.output_dims,
                            self.action_space,
                            memory=None,
                            load_model=agent_policy,
                            **self.config)
            agents[_id] = agent
        return agents


if __name__ == "__main__":
    try:
        env = Game(config)
    except: 
        print(f"Failed to initialize the Game envrironment!")
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    action_space = env.action_space
    evaluate = Evaluate(config, env, 
            input_dims = input_dims,
            output_dims = output_dims,
            action_space = action_space)
    evaluate.evaluate()
