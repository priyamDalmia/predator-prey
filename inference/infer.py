import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pdb
import matplotlib.pyplot as plt
from data.helpers import dodict
from data.common import ACTIONS
from agents.tor_par_AC import AACAgent
import argparse
import yaml

parser = argparse.ArgumentParser(description="experiments",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--id', type=int, help="The configuration to run")
ARGS = parser.parse_args()
'''
NOTE: This is not a simulation of the game state evolution. Real dynamics of the 
game are not implemented here. The purpose of this is to only generate psudo-observations 
to collect data.

Giving starting positions and a list of actions takens, an observation for a 
"simulated" game can be generated. 

Make sure that the list of actions is a valid or "premissible" list of actions
otherwise, the observations recived will not be a reflection of the real game.
'''
## If GAME size chagnes (from 10) - ADJUST @ must be modified. 
ADJUST = lambda x, y: (x[0]+y, x[1]+y)
ADJUST_2 = lambda x, y, z: (y-(z[0]-x[0]), y-(z[1]-x[1]))
RANGE = lambda x, y, z: ((max(0, x[0]-y), min(x[0]+y+1, z)), (max(0, x[1]-y), min(x[1]+y+1, z)))
COLOR = []

config = dict(
        # Environment size, 
        size=10,
        winsize=9,
        npred=2,   # Be Careful when setting these values.
        nprey=2,
        pred_pos=[(3,3), (5,5)],
        prey_pos=[(1,1), (3,4)],
        target_pos=[(5,5)], # Should be one of the prey positons.
        agent_class = AACAgent,
        agent_policy = "experiments/common/pred1",
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
        

    def run_inference(self):
        self.target_pos = self.config.target_pos[0]
        target_id, idx = self.initialize_agents(self.target_pos)
        self.build_game_state()
        self.render()
        agent_state = self.create_base_observation(idx, target_id, self.target_pos)
        # GET VALUES AND ACTION PROBS
        for i, x in enumerate(self.X[0,:]):
            for j, y in enumerate(self.Y[:,0]):
                # get observation for _id
                # !!position = self.all_pos[_id]
                position = (x, y)
                #self.update_state(idx+1, position)
                observation = self.get_base_state(agent_state.copy(), idx+1, position)
                # Make Inference. Get Q values.
                probs, values = self.infer_agents[target_id].get_raw_output(observation)   
                # store the obtainded valeus 
                self.store_values(target_id, position, (j ,i), probs.tolist(), values) 
        # GERNERATE THE PLOTS
        self.vec_field_graph()
    
    def create_base_observation(self, target_idx, target_id, target_pos):
        observation = self.get_observation(target_idx+1, target_pos)
        channel_0 =  np.pad(observation[0], pad_width=self.pad_width, mode="edge")
        channel_1 = np.pad(observation[1], pad_width=self.pad_width, mode="constant")
        channel_2 = np.pad(observation[2], pad_width=self.pad_width, mode="constant")
        channel_1[(2*self.pad_width) ,(2*self.pad_width)] = 0
        return np.stack((channel_0, channel_1, channel_2))
    
    def get_base_state(self, agent_state, idx, position):
        z = self.target_pos        
        (pos_x, pos_y) = ADJUST_2(position, (2*self.pad_width), z)
        observation = agent_state[:,
                pos_x-self.pad_width:pos_x+self.pad_width+1,
                pos_y-self.pad_width:pos_y+self.pad_width+1]
        observation[1, self.pad_width, self.pad_width] = 1
        return observation

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
        
    def store_values(self, _id, position, arr_idx, probs, values):
        probs = probs[0]
        if position in self.config.prey_pos:
            color = [1]
            probs = [0.0, 0.0, 0.0, 0.0]
        elif position in self.config.pred_pos:
            color = [2]
            probs = [0.0, 0.0, 0.0, 0.0]
        else:
            color = [0]
        data = probs + [values.item()] + color
        try:
            self.infer_values[_id][:, arr_idx[0], arr_idx[1]] = data
        except Exception as e:
            print(f"Array index out of bounds! Fix array dimensions for: {_id}")

    def initialize_agents(self, target_pos):
        # Go Over All predators 
        # Create a list of Agents ids, Agent position, and Inference Policies.
        # Create Empty array to store inference values. 
        # For Each Agent (4 prob + 1 value functions)
        # The prey around which to infer.
        self.target_pos = target_pos
        target_id = None
        x_lim, y_lim = RANGE(self.target_pos, self.pad_width, self.config.size)
        # Creating a numpy meshgrid for all points.
        x_range = np.arange(x_lim[0], x_lim[1])
        y_range = np.arange(y_lim[0], y_lim[1])
        self.X, self.Y = np.meshgrid(x_range, y_range)
        self.plot_X, self.plot_Y = np.meshgrid(\
                range(len(x_range)), range(len(y_range)))
        self.infer_array = np.zeros((6, len(x_range), len(y_range)), dtype=np.object_)

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
            if position == target_pos:
                infer_agents[_id] = agent
                infer_values[_id] = self.infer_array.copy()
                target_id = _id
                idx = int(target_id[-1])
        for i in range(self.config["nprey"]):
            _id = f"prey_{i}"
            all_agents.append(_id)
            position = self.config["prey_pos"][i]
            all_pos[_id] = position
        self.all_agents = all_agents
        self.all_pos = all_pos
        self.infer_agents = infer_agents
        self.infer_values = infer_values
        if not target_id:
            raise("Target ID could not be initialized! Fix the target and restart.")
        return target_id, idx

    def vec_field_graph(self):
        for _id in self.infer_agents:
            data_i = self.infer_values[_id]
            mat_X = data_i[1,:,:] - data_i[0,:,:]
            mat_Y = data_i[2,:,:] - data_i[3,:,:]
            mat_X = np.array(mat_X, dtype=np.float32)
            mat_Y = np.array(mat_Y, dtype=np.float32)
            color = np.array(data_i[-1,:,:], dtype=np.float32)
            plt.quiver(self.Y, -self.X, -mat_X, -mat_Y, color, pivot='mid', units='width')
            ax = plt.gca()
            ax.set_xticks([x-0.5 for x in self.X[1,:]],minor=True)
            ax.set_yticks([-x-0.5 for x in self.Y[:,1]],minor=True)
            plt.grid(which="minor", ls="--",lw=1, alpha=0.5)
            for n, pos in enumerate(self.config.pred_pos):
                plt.text(pos[0], pos[1], f"T{n}")
            for n, pos in enumerate(self.config.prey_pos):
                plt.text(pos[0], pos[1], f"D{n}")
            breakpoint()
            plt.show()
            plt.savefig(f"{self.config.plot_file}_{_id}")
            print(f"Plot Saved: {_id} -> {self.config.plot_file}_{_id}")
        pass

    def game_state_graph(self):
        breakpoint()
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
    config = config
    infer_id = ARGS.id
    for i in range(infer_id):
        with open('inference/config.yaml') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            config.update(config_data["infer"]["global"])
            config.update(config_data["infer"][f"sce_{i+1}"])
    infer = Inference(config)
    infer.run_inference()
