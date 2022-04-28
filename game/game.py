import os
import logging
import numpy as np
import random
import time
import json
from data.common import ACTIONS
from game.game_state import GameState

import pdb

class Game():
    def __init__(self, config):
        self.size = config.size
        self.npredators = config.npred
        self.npreys = config.nprey
        self.window_size = config.winsize
        self.pad_width = int(self.window_size/2)
        self.game_state = None

        # Game managment variables
        self.units = (self.npredators + self.npreys + 1)
        self.action_space = [i for i in range(4)]
        self.observation_space = np.zeros((3, self.window_size, self.window_size), dtype=np.int32)
        self.state_space = np.zeros((self.units, self.size, self.size), dtype=np.int32)
        self.last_action = None

        self.record = {}
        
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
        self.done = {}
        # Create a new GameState Object
        del self.game_state
        self.game_state = GameState(self.size, self.npredators, self.npreys, self.window_size, self.pad_width)
        
        # Episode records 
        self.steps = 0 
        self.record = {}
        # Populate Predators
        for i in range(self.npredators):
            _id = f"predator_{i}"
            agent_obj , position = self.game_state.add_unit(_id)
            self.agents[_id] = (len(self.agent_ids)+1, agent_obj, position)
            self.agent_ids.append(_id)
            self.last_rewards[_id] = 0
            self.predator_pos[_id] = position
            self.pos_predator[position] = _id
            self.done[_id] = False
        # Populate Preys 
        for i in range(self.npreys):
            _id = f"prey_{i}"
            agent_obj, position = self.game_state.add_unit(_id)
            self.agents[_id] = (len(self.agent_ids)+1, agent_obj, position)
            self.agent_ids.append(_id)       
            self.last_rewards[_id] = 0
            self.prey_pos[_id] = position
            self.pos_prey[position] = _id
            self.done[_id] = False
        # returns the initial observation dict.
        self.record_transition(0, 0, 0)
        return dict(self.get_observation()), self.done
    
    def step(self, actions:dict) -> tuple:
        # takes an action dict and updates the underlying game state.
        # step over the action list and get new positions.
        # 1. No overlaps are allowed here.
        # ! Modify for any changes in the reward structure here !
     
        if len(actions)!=len(self.agent_ids):
            raise print("Error: action sequence of incorrect length!")
            
        rewards = dict(self.last_rewards)
        action_ids = list(actions.keys())
        random.shuffle(action_ids)
        for _id in action_ids:
            action = actions[_id]
            # Predator Action execution
            if _id.startswith("predator"):
                position = self.predator_pos[_id]
                if not position:
                    # Agent has died in this turn.
                    # Remove all agent properties.
                    continue
                new_position = ACTIONS[action](position, self.size, self.pad_width)
                if new_position == position:
                    rewards[_id] += -0.01
                else:
                    # check collison with mates and stop update
                    if self.game_state.predator_collision(*new_position):
                        new_position = position
                        rewards[_id] -= -0.01
                    elif self.pos_prey.get(new_position):
                        a = self.pos_prey.get(new_position)
                        rewards[a] -= 1
                        rewards[_id] += 1
                        # Remove the positon all together!!
                        self.prey_pos[a] = (0, 0)
                        self.done[a] = True
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
                    rewards[_id] = -1
                    self.done[_id] = True
                    # Prey has died. Remove all traces of the prey from the environment.
                    # or Respawn.
                    continue
                new_position = ACTIONS[action](position, self.size, self.pad_width)
                if new_position == position:
                    rewards[_id] += -0.01
                else:
                    # check collision with mates
                    if self.game_state.prey_collision(*new_position):
                        new_position = position
                    elif self.pos_predator.get(new_position):
                        a = self.pos_predator.get(new_position)
                        rewards[a] += 1
                        rewards[_id] -= 1
                        new_position = (0, 0)
                        self.done[_id] = True
                del self.pos_prey[position]
                self.prey_pos[_id] = new_position
                self.pos_prey[new_position] = _id
            # Update the state in the game_state obj.
            self.game_state.update_unit(self.agents[_id][0], new_position)
        info = {}
        self.steps+=1
        self.record_transition(actions, rewards, self.done)
        done = True if sum(self.done.values())==self.npreys else False
        return rewards, self.get_observation(), done, info

    def get_observation(self) -> dict:
        observation = {}
        idx = 1
        for _id, position in self.predator_pos.items():
            observation[_id] = self.game_state.observe(_id, idx, *position)
            idx += 1
        for _id, position in self.prey_pos.items():
            observation[_id] = self.game_state.observe(_id, idx, *position)
            idx += 1
        return dict(observation)

    def render(self, mode="human"):
        adjust = lambda x, y: (x[0]-y, x[1]-y)
        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
            
        for _id, position in self.predator_pos.items():
            (x, y) = adjust(position, self.pad_width)
            gmap[x][y]  = f"T{_id[-1]}"
        for _id, position in self.prey_pos.items():
            if position == (0,0):
                continue
            (x, y) = adjust(position, self.pad_width)
            gmap[x][y]  = f"D{_id[-1]}"
        
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]
        print(np.matrix(gmap))
    
    def record_transition(self, actions, rewards, done):
        transition = {}
        transition['actions'] = str(actions)
        transition['rewards'] = str(rewards)
        transition['done'] = str(done)
        transition['pred_pos'] = str(self.predator_pos)
        transition['prey_pos'] = str(self.prey_pos)
        self.record[str(self.steps)] = transition

    def record_episode(self, filename, info={}):
        game_data = {}
        game_data['size'] = self.size
        game_data['pad_width'] = self.pad_width 
        game_data['units'] = self.units
        game_data['ep_record'] = self.record
        #game_data['info'] = info 
        with open(f"replays/{filename}", 'w') as f:
            json.dump(game_data, f)
