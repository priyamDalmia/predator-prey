from re import I
import numpy as np
import json 
import argparse
import ast
import sys 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = "Predator_Prey Renderer.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('filename', type=str, metavar="", help="json file to be rendered")
parser.add_argument('-t', type=bool, metavar="", help="Render in the Terminal or Create a Matplotlib animation.", default=True)
ADJUST = lambda x, y: (x[0]-y, x[1]-y)

class Renderer():
    def __init__(self, filename, t):
        with open(filename) as f:
            self.game_data = json.load(f)
        self.size = self.game_data['size']
        self.pad_width = self.game_data['pad_width']
        self.units = self.game_data['units']
        self.record = self.game_data['ep_record']
        self.t = t
        #self.info = info

    def replay(self):
        if self.t:
            for _step in self.record.keys():
                pred_pos = self.record[_step]["pred_pos"]
                pred_pos = ast.literal_eval(pred_pos)
                prey_pos = self.record[_step]["prey_pos"]
                prey_pos = ast.literal_eval(prey_pos)
                print(f"ACTIONS: {self.record[_step]['actions']}")
                print(f"REWARDS: {self.record[_step]['rewards']}")
                print(f"STEPS: {_step}")
                if _step == '0':
                    self.render(pred_pos, prey_pos)
                    key = input("Press Enter to START")
                    self.clear_lines((self.size+12))
                else:
                    self.render(pred_pos, prey_pos)
                    key = input("press Enter to continue")
                    self.clear_lines((self.size+12))
                if key == "e":
                    sys.exit()

    def render_terminal(self, predator_pos, prey_pos):
         gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
         for _id, position in predator_pos.items():
             (x, y) = ADJUST(position, self.pad_width)
             gmap[x][y]  = f"T{_id[-1]}"
         for _id, position in prey_pos.items():
             if position == (0,0):
                 continue
             (x, y) = ADJUST(position, self.pad_width)
             gmap[x][y]  = f"D{_id[-1]}"

         gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]
         print(np.matrix(gmap))

    def clear_lines(self, lines=15):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2k'
        for i in range(lines):
            print(LINE_CLEAR, end=LINE_UP)
        
    def render_plot(self):
        fig, ax = plt.subplots()
        ax.set_title("Predator-Prey Environment")
        # Set the limits of the axis
        ax.set_xlim(0, self.size)
        ax.set_ylim(self.size,0)
        ax.set_xticks(range(0,self.size))
        ax.set_yticks(range(0,self.size)) 
        # Create the grid and mark the positions
        ax.grid(alpha=0.2)    
        X , Y = np.meshgrid(range(0, self.size), range(0, self.size))
        ax.scatter(X+0.5, Y+0.5, color="black", alpha=0.1, s=4)
           

if __name__=="__main__":
    args = parser.parse_args()
    filename = f"./{args.filename}"
    renderer = Renderer(filename, args.t)
    renderer.replay()
