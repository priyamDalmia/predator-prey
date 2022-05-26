import ast
import numpy as np
import json
import argparse
import ast
import sys


parser = argparse.ArgumentParser(description = "Predator_Prey Renderer.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('filename', type=str, metavar="", help="json file to be rendered")
parser.add_argument('-t', type=bool, metavar="", help="Render in the Terminal or Create a Matplotlib animation.", default=True)
ADJUST = lambda x, y: (x[0]-y, x[1]-y)



class Renderer(self):
    def __init__(self, filename,):
        with open(filename) as f:
            self.game_data = json.load(f)
        self.size = self.game_data['size']
        self.pad_width = self.game_data['pad_width']
        self.units = self.game_data['units']
        self.record = self.game_data['ep_record']
        self.steps = len(self.record)

    def render_plot(self):
        breakpoint()
        fig, ax = plt.subplots()
        # Setting axis range and ticks
        ax.set_xlim(0, self.size)
        ax.set_ylim(self.size, 0)
        ax.set_xticks(range(0, self.size))
        ax.set_yticks(range(0, self.size))
        # Creating the grids
        ax.grid(alpha=0.5)
        ax.set_title("Predator-Prey Environemt")
        X, Y = np.meshgrid(range(0, self.size), range(0, self.size))
        sca = ax.scatter(X+0.5, Y+0.5, color="black", alpha=0.1, s=4)

        def animate(i):
            breakpoint()
            pred_pos, prey_pos = self.get_data_tstep(i)
             

    def get_data_tstep(self, timestep):
        k = str(timestep)
        pred_pos = self.record[step]["pred_pos"]
        pred_pos = ast.literal_eval(pred_pos)
        prey_pos = self.record[step]["prey_pos"]
        prey_pos = ast.literal_eval(prey_pos)
        return pred_pos, prey_pos

if __name__ == "__main__"
    args = parser.parse_args()
    filename = f"./{args.filename}"
    renderer = Renderer(filename)
    renderer.replay()
