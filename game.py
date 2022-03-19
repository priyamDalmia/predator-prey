import os
import logging
import numpy as np
import random
import time

from data.entites import Predator, Prey
from data.common import ACTIONS
from game_state import GameState


class Game():
    def __init__(self, args):
        self.size = args.size
        self.npredators = args.npred
        self.npreys = args.nprey
        self.window_size = args.win_size
        self.pad_width = int(args.win_size/2)
        self.game_state = None

        # Game managment variables
        self.units = (self.npredators + self.npreys + 1)
        self.action_space = [i for i in range(4)]
        self.observation_space = np.zeros((self.units, self.size, self.size), dtype=np.int32)
        self.last_action = None
        
        # Reset the environment object (basically recreates a new GameState object.)
        self.reset()   

    
    def reset(self) -> dict:
        self.agents = {}
        self.agent_ids = []
        self.last_rewards = {}
        self.predator_pos = {}
        self.prey_pos = {}
        # A inverse hash list
        self.pos_predator = {}
        self.pos_prey = {}
        # Create a new GameState Object
        self.game_state = GameState(self.size, self.npredators, self.npreys, self.window_size, self.pad_width)

        # Populate Predators
        for i in range(self.npredators):
            _id = f"predator_{i}"
            agent_obj , position = self.game_state.add_unit(_id)
            self.agents[_id] = (len(self.agent_ids)+1, agent_obj, position)
            self.agent_ids.append(_id)
            self.last_rewards[_id] = 0
            self.predator_pos[_id] = position
            self.pos_predator[position] = _id
        # Populate Preys 
        for i in range(self.npreys):
            _id = f"prey_{i}"
            agent_obj, position = self.game_state.add_unit(_id)
            self.agents[_id] = (len(self.agent_ids)+1, agent_obj, position)
            self.agent_ids.append(_id)       
            self.last_rewards[_id] = 0
            self.prey_pos[_id] = position
            self.pos_prey[position] = _id
        # returns the initial observation dict.
        return self.get_observation()

    def step(self, actions:dict) -> tuple:
        # takes an action dict and updates the underlying game state.
        # step over the action list and get new positions.
        # 1. No overlaps are allowed here.
        # 2. Improve efficieny using reassignemnt instead of recreation.
        # ! Modify for any changes in the reward structure here !
     
        if len(actions)!=len(self.agent_ids):
            raise print("Error: action sequence of incorrect length!")

        rewards = self.last_rewards

        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        action_ids = list(actions.keys())
        random.shuffle(action_ids)
        
        for _id in action_ids:
            action = actions[_id]
            rewards[_id] = 0
            # Predator Action execution
            if _id.startswith("predator"):
                position = self.predator_pos[_id]
                if not position:
                    # Agent has died in this turn.
                    # Remove all agnet properties.
                    break
                new_position = ACTIONS[action](position, self.size, self.pad_width)
                if new_position == position:
                    rewards[_id] += -1
                else:
                    # check collison with mates and stop update
                    if self.game_state.predator_collision(*new_position):
                        new_position = position
                        reward[_id] = 0 
                    elif a := self.pos_prey.get(new_position):
                        rewards[a] -= 10
                        rewards[_id] += 10
                        # Remove the positon all together!!
                        self.prey_pos[a] = (0, 0)
                        del self.pos_prey[new_position]
                del self.pos_predator[position]
                self.predator_pos[_id] = new_position
                self.pos_predator[new_position] = _id
            # Prey Action execution.                
            else:
                position = self.prey_pos[_id]
                if position == (0, 0):
                    new_position = position
                    self.game_state.update_unit(self.agents[_id][0], new_position)
                    # Prey has died. Remove all traces of the prey from the environment.
                    # or Respawn.
                    break
                new_position = ACTIONS[action](position, self.size, self.pad_width)
                if new_position == position:
                    rewards[_id] += -1
                else:
                    # check collision with mates
                    if self.game_state.prey_collision(*new_position):
                        new_position = position
                        reward[_id] = 0
                    if a := self.pos_predator.get(new_position):
                        rewards[a] += 10
                        rewards[_id] -= 10
                        new_position = (0, 0)
                try:
                    del self.pos_prey[position]
                    self.prey_pos[_id] = new_position
                    self.pos_prey[new_position] = _id
                except Exception as e:
                    print(e)
                    breakpoint()

            # Update the state in the game_state obj.
            self.game_state.update_unit(self.agents[_id][0], new_position)
        
        done = False
        info = {}
        return rewards, self.get_observation, done, info

    def get_observation(self) -> dict:
        observation = {}
        for _id, value in self.agents.items():
            observation[_id] = self.game_state.observe(_id, *value)
        self.observation = {}
        return observation

    def render(self, mode="human"):
    
        adjust = lambda x, y: (x[0]-y, x[1]-y)
        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
        for _id, position in self.predator_pos.items():
            (x, y) = adjust(position, self.pad_width)
            gmap[x][y]  = f"T{_id[-1]}"
        for _id, position in self.prey_pos.items():
            if position == (0,0):
                break
            (x, y) = adjust(position, self.pad_width)
            gmap[x][y]  = f"D{_id[-1]}"
        
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]
        print(np.matrix(gmap))


