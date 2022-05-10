import os
import sys
import numpy as np
import pdb

'''
NOTE: This is not a simulation of the game state evolution. Real dynamics of the 
game are not implemented here. The purpose of this is to only generate psudo-observations 
to collect data.

Giving starting positions and a list of actions takens, an observation for a 
"simulated" game can be generated. 

Make sure that the list of actions is a valid or "premissible" list of actions
otherwise, the observations recived will not be a reflection of the real game.
'''

ADJUST = lambda x, y: (x[0]+y, x[1]+y)

config = dict(
        # Environment size, 
        size=7,
        winsize=5,
        npred=2,
        nprey=2,
        pred_pos=[(0,1), (1,1)],
        prey_pos=[(3,4), (2,2)],
        )

class Inference():
    def __init__(self, config):
        self.config = config    
        self.initialize_agents()
        self.build_game_state()

    def run_inference(self):
        while True:
            breakpoint()
            for idx, _id in enumerate(self.infer_agents.keys()):
                # get observation for _id
                position = self.all_pos[_id]
                observation = self.get_observation(idx+1, position)
                # Make Inference. Get Q values.
                # values = self.agents[_id].forward()
                
                # update the posititon of the agents
                new_position = self.update_position(_id, position) 
                self.all_pos[_id] = new_position
                self.update_state(idx+1, new_position)
        pass

    def build_game_state(self):
        # The Base Game State.
        self.size = self.config["size"]
        self.win_size = self.config["winsize"]
        self.pad_width = int(self.win_size/2)
        base_state = np.pad((np.zeros((self.size, self.size), dtype=np.int32)),\
                pad_width=self.pad_width, constant_values=1)
        self.state = np.expand_dims(base_state, axis=0).copy()
        self.channel = np.pad((np.zeros((self.size, self.size), dtype=np.int32)),\
                pad_width=self.pad_width, constant_values=0)
        # For each agent add their positions and build new game states.
        for _id in self.all_agents:
            # The postions of each agent.
            position = self.all_pos[_id]
            position = ADJUST(position, self.pad_width)
            new_channel = self.channel.copy()
            new_channel[position[0], position[1]] = 1
            self.state = np.vstack((self.state, np.expand_dims(new_channel, axis=0)))

    def update_state(self, idx, new_position):
        # Adjust postion for padding.
        self.state[idx, :, :] = self.channel.copy()
        self.state[idx, new_position[0], new_position[1]] = 1

    def get_observation(self, idx, position):
        # Adjust position for padding.
        (pos_x, pos_y) = ADJUST(position, self.pad_width)
        observation = self.state[:,
                pos_x-self.pad_width:pos_x+self.pad_width+1,
                pos_y-self.pad_width:pos_y+self.pad_width+1]
        channel_0 = observation[0]
        channel_1 = np.sum(observation[1:self.npreds+1], axis=0)
        channel_2 = np.sum(observation[self.npreds+1:], axis=0)
        return np.stack((channel_0, channel_1, channel_2))
        
    def render(self):
        gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
        for _id, position in self.all_pos.items():
            if _id.startswith("pred"):
                str_id = f"T{_id[-1]}"
            else:
                str_id = f"D{_id[-1]}"
            gmap[position[0]][position[1]] = str_id
        gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]
        print(np.matrix(gmap))

    def initialize_agents(self):
        # Go Over All predators 
        # Create a list of Agents ids, Agent position, and Inference Policies.
        self.npreds = self.config['npred']
        self.npreys = self.config['nprey']
        all_agents = []
        all_pos = {}
        for i in range(self.config["npred"]):
            _id = f"predator_{i}"
            all_agents.append(_id)
            position = self.config["pred_pos"][i]
            all_pos[_id] = position 
            # Load a policy or Network here.
            # Add object to infer policy
            pass
        for i in range(self.config["nprey"]):
            _id = f"prey_{i}"
            all_agents.append(_id)
            position = self.config["prey_pos"][i]
            all_pos[_id] = position 
        self.all_agents = all_agents
        self.all_pos = all_pos
if __name__ == "__main__":
    breakpoint()
    infer = Inference(config)
