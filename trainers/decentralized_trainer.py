import sys
from typing import List, Optional
import copy
from trainers.utils import *
from trainers.trainer import Trainer
from data.config import TrainerConfig
from examples.tiger_deer import Game
from agents.agent import BaseAgent

class DecentralizedTrainer(Trainer):
    def __init__(self, 
            config: TrainerConfig, 
            game: Game,
            agent_ids: List,
            agents: Optional[BaseAgent]= None): 
        super().__init__(config, agent_ids)
        self.game = game

        
        # initialize agents 
        if agents:
            self.agent_policies = agents
        else:
            # Techincally config should contain the mode for agent inti
            self.agent_init = config.agent_init_func
            self.agent_policies = self.agent_init(
                game.num_agents,
                game.agent_ids,
                game.observation_space(),
                game.action_space())

        # reset game 

        pass
        
   # TODO create wrapper to send data back and forth
    def run_episodes(self, num_games: Optional[int]):
        game = copy.deepcopy(self.game)
        for episode in range(num_games):
            game.reset()
            done = game.is_terminal()
            game_steps = 0
            while not done:
                actions = {}
                # Pre move operations - message, reward, consensus etc
                self.pre_action_ops()
                for agent_id, agent in self.agent_policies.items():
                    # TODO check if agent is in game agents and then choose action 
                    agent_observation = game.get_observation(agent_id)
                    action = agent.get_action(agent_observation)
                    actions[agent_id] = action
                # assert that all actions are present
                # TODO for any agent, check that action is not out of bounds
                game_steps += 1
                game.step(actions)
                done = game.is_terminal()
                # check training type
                # and save experience to buffer in accordance. 
                self.post_action_ops()

    def pre_action_ops(self):
        pass
        
    def post_action_ops(self):
        pass

