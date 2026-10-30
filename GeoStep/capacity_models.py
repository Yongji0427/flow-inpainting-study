"""Width-only control; forward computation is inherited from the frozen pilot."""
import math
import torch
from torch import nn
from pilot import UNet, Block


class WideUNet(UNet):
    """Double all spatial channels; retain the original depth/time embedding."""
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer('freq', torch.exp(torch.linspace(0,-math.log(10000),32)))
        self.temb = nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128))
        self.inp = nn.Conv2d(7,64,3,padding=1)
        self.e0,self.e1,self.e2 = Block(64,64),Block(128,128),Block(192,192)
        self.d0,self.d1,self.d2 = nn.Conv2d(64,128,3,2,1),nn.Conv2d(128,192,3,2,1),nn.Conv2d(192,256,3,2,1)
        self.mid = nn.ModuleList([Block(256,256),Block(256,256)])
        self.u2,self.u1,self.u0 = nn.Conv2d(256,192,3,padding=1),nn.Conv2d(192,128,3,padding=1),nn.Conv2d(128,64,3,padding=1)
        self.b2,self.b1,self.b0 = Block(384,192),Block(256,128),Block(128,64)
        self.out = nn.Sequential(nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,3,3,padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
