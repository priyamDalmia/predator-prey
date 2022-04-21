import numpy as np
import json 
import argparse
import ast
import sys 

parser = argparse.ArgumentParser(description = "Predator_Prey Renderer.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('filename', type=str, metavar="", help="json file to be rendered")
parser.add_argument('-t', type=str, metavar="", help="Frame rate (time sleep b/w frames)", default=0.2)

class Renderer():
    def __init__(self, filename, time_sleep):
        with open(filename) as f:
            self.game_data = json.load(f)
        self.size = self.game_data['size']
        self.pad_width = self.game_data['pad_width']
        self.units = self.game_data['units']
        self.record = self.game_data['ep_record']
        #self.info = info

    def replay(self):
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
                self.clear_lines((self.size+4))
            else:
                self.render(pred_pos, prey_pos)
                key = input("press Enter to continue")
                self.clear_lines((self.size+4))
            if key == "e":
                sys.exit()

    def render(self, predator_pos, prey_pos):
         adjust = lambda x, y: (x[0]-y, x[1]-y)
         gmap = np.zeros((self.size, self.size), dtype=np.int32).tolist()
         for _id, position in predator_pos.items():
             (x, y) = adjust(position, self.pad_width)
             gmap[x][y]  = f"T{_id[-1]}"
         for _id, position in prey_pos.items():
             if position == (0,0):
                 continue
             (x, y) = adjust(position, self.pad_width)
             gmap[x][y]  = f"D{_id[-1]}"

         gmap = [list(map(lambda x: "." if x == 0 else x, l)) for l in gmap]
         print(np.matrix(gmap))

    def clear_lines(self, lines=1):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2k'
        for i in range(lines):
            print(LINE_CLEAR, end=LINE_UP)

if __name__=="__main__":
    args = parser.parse_args()
    filename = f"./replays/{args.filename}"
    renderer = Renderer(filename, args.t)
    renderer.replay()
