import os
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from data.helpers import dodict
from data.common import ACTIONS
from agents.tor_naac import AACAgent
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
RANGE = lambda x, y: ((0, x[0]+y+1), (0, x[1]+y+1))

config = dict(
        # Environment size, 
        size=15,
        winsize=9,
        npred=2,   # Be Careful when setting these values.
        nprey=2,
        pred_pos=[(3,3), (2,4)],
        prey_pos=[(1,1), (3,3)],
        target_pos=[(1,1)], # Should be one of the prey positons.
        agent_class = AACAgent,
        agent_policy = "experiments/2/policies/predator_0-15-2ac-2rand-2399-48",
        critc_class = "",
        critic_policy = "",
        plot_file="plots/plot_test",
        steps=5,
        )

class Inference():
    def __init__(self, config):
        self.config = dodict(config) 
        self.units = self.config["npred"] + self.config["nprey"]
        self.input_dims = (3, self.config["winsize"], self.config["winsize"])
        self.output_dims = 4
        self.action_space = [i for i in range(4)]
        self.size = self.config["size"]
        self.win_size = self.config["winsize"]
        self.pad_width = int(self.win_size/2)
        
        self.initialize_agents()
        self.build_game_state()

    def run_inference(self):
        for x in range(len(self.X)):
            for y in range(len(self.Y)):
                for idx, _id in enumerate(self.infer_agents.keys()):
                    # get observation for _id
                    # !!position = self.all_pos[_id]
                    position = (x, y)
                    self.all_pos[_id] = position
                    self.update_state(idx+1, position)
                    observation = self.get_observation(idx+1, position)
                    # Make Inference. Get Q values.
                    probs, values = self.infer_agents[_id].get_raw_output(observation)   
                    # store the obtainded valeus 
                    self.store_values(_id, position, probs, values) 
                    if _id.startswith("critic"): 
                        continue
                    # update the posititon of the agents
    
    def build_game_state(self):
        # The Base Game State.
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
        new_position = ADJUST(new_position, self.pad_width)
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
        
    def store_values(self, _id, position, probs, values):
        if position in self.config.prey_pos:
            return
        data = probs.tolist()[0] + values.tolist()[0]
        self.infer_values[_id][:, position[0], position[1]] = data

    def initialize_agents(self):
        # Go Over All predators 
        # Create a list of Agents ids, Agent position, and Inference Policies.
        # Create Empty array to store inference values. 
        # For Each Agent (4 prob + 1 value functions)
        # The prey around which to infer.
        self.target_pos = self.config.target_pos[0]
        x_lim, y_lim = RANGE(self.target_pos, self.pad_width)
        # Creating a numpy meshgrid for all points.
        x_range = np.arange(x_lim[0], x_lim[1])
        y_range = np.arange(y_lim[0], y_lim[1])
        self.Y, self.X = np.meshgrid(x_range, y_range)
        self.infer_array = np.zeros((5, len(x_range), len(y_range)))

        self.npreds = self.config['npred']
        self.npreys = self.config['nprey']
        all_agents = []
        all_pos = {}
        infer_agents = {}
        infer_values = {}
        for i in range(self.config["npred"]):
            _id = f"predator_{i}"
            all_agents.append(_id)
            position = self.config["pred_pos"][i]
            all_pos[_id] = position 
            # Write code to add Policy(actors) and Critics seperately.
            agent = self.config["agent_class"](
                    _id, 
                    self.input_dims,
                    self.output_dims,
                    self.action_space,
                    memory=None, 
                    load_model = self.config.agent_policy,
                    eval_model = True,
                    )
            infer_agents[_id] = agent
            infer_values[_id] = self.infer_array.copy()
        for i in range(self.config["nprey"]):
            _id = f"prey_{i}"
            all_agents.append(_id)
            position = self.config["prey_pos"][i]
            all_pos[_id] = position 
        self.all_agents = all_agents
        self.all_pos = all_pos
        self.infer_agents = infer_agents
        self.infer_values = infer_values

    def vec_field_graph(self):
        breakpoint()
        for _id in self.infer_agents:
            data_i = self.infer_values[_id]
            mat_X = data_i[0,:,:] - data_i[1,:,:]
            mat_Y = data_i[2,:,:] - data_i[3,:,:]
            plt.quiver(self.X, self.Y, mat_X, mat_Y)
            plt.savefig(f"{self.config.plot_file}_{_id}")
        pass
        
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

    def next_position(self, _id, position):
        next_action = None
        new_position = ACTIONS[action](position, self.size, self.pad_width)
        return new_position

if __name__ == "__main__":
    infer = Inference(config)
    infer.run_inference()

    # Make Vector Feild Graph
    infer.vec_field_graph()
