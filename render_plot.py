import numpy as np
import sys 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pdb
import matplotlib
from collections import deque

from renderer import Renderer

ADJUST = lambda x, y: (x[0]-y, x[1]-y)

class RenderPlot():
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.renderer = Renderer(filename, 0.1)
        self.size = self.renderer.size
        self.pad_width = self.renderer.pad_width
        self.record = self.renderer.record
        self.units = self.renderer.units

    def create_plot(self, ptype, sf=None):
        if sf:
            steps = sf
        else:
            steps = len(self.renderer.record)

        num_agents = self.units
        size = self.size
        if ptype == "simple":
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, size)
            self.ax.set_ylim(size, 0)
            self.ax.set_xticks(range(0, size))
            self.ax.set_yticks(range(0, size))
        self.ax.grid(alpha=0.2)
        self.ax.set_title("Predator-Prey Environment")

        X , Y = np.meshgrid(range(0, size), range(0, size))
        self.ax.scatter(X+0.5, Y+0.5, color="black", alpha=0.1, s=3)
        sc_pred = self.ax.scatter([], [], label="Predator", \
                color='red', edgecolors='k', marker="o")
        sc_prey= self.ax.scatter([], [], label="Prey", \
                color="green", edgecolors='k', marker = "X")
        
        breakpoint()
        lines = []
        l_points = []
        rects = []
        for i in range(num_agents):
            lobj, = self.ax.plot([], [], lw=1)
            lines.append(lobj)
            l_points([0, 0])
            rect = matplotlib.Rectangle((1,1), width=5, height=5)
            rects.append(rects)

    def init_plot():
        pass

    def animate():
        pass

    def 


if __name__ === "__main__":
    filename = f"./inference/5/20-.."
    renderer = RenderPlot(filename)
