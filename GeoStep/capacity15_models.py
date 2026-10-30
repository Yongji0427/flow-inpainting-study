"""Width-only control; forward computation is inherited from the frozen pilot."""
import math
import torch
from torch import nn
from pilot import UNet, Block


class LargeUNet(UNet):
    """Triple all spatial channels; retain the original depth/time embedding."""
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer('freq', torch.exp(torch.linspace(0,-math.log(10000),32)))
        self.temb = nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128))
        self.inp = nn.Conv2d(7,96,3,padding=1)
        self.e0,self.e1,self.e2 = Block(96,96),Block(192,192),Block(288,288)
        self.d0,self.d1,self.d2 = nn.Conv2d(96,192,3,2,1),nn.Conv2d(192,288,3,2,1),nn.Conv2d(288,384,3,2,1)
        self.mid = nn.ModuleList([Block(384,384),Block(384,384)])
        self.u2,self.u1,self.u0 = nn.Conv2d(384,288,3,padding=1),nn.Conv2d(288,192,3,padding=1),nn.Conv2d(192,96,3,padding=1)
        self.b2,self.b1,self.b0 = Block(576,288),Block(384,192),Block(192,96)
        self.out = nn.Sequential(nn.GroupNorm(8,96),nn.SiLU(),nn.Conv2d(96,3,3,padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
