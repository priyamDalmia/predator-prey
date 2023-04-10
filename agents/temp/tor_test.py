import os
import numpy as np
import pdb

import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, inp, out):
        super(Network, self).__init__()
        self.inp = inp
        self.out = out

if __name__ == "__main__":
    print("Torch network")
    
    breakpoint()

    x = torch.rand(2, requires_grad=True)
    #w = torch.Tensor([2.0, 2.0, 2.0, 2.0])
    print(x)

    y = x+2

    z = y*y*2

    z.backward()

    breakpoint()



    
