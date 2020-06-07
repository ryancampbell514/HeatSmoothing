import torch as th
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .utils import View, Avg2d

class AllCNNBase(nn.Module):
    def __init__(self,  conv, bn=True, kernel_size=(3,3), 
            in_channels=3, classes=10, c1=96, c2=192,
            nonlinear='relu', softmax=False, dropout=0.,bias=True, 
            last_layer_nonlinear=False, last_layer_bn=None, **kwargs):
        """Implementation of AllCNN-C [1].

           [1] Springenberg JT, Dosovitskiy A, Brox T, Riedmiller M. Striving
               for simplicity: The all convolutional net. arXiv preprint
               arXiv:1412.6806.  2014 Dec 21."""
        if last_layer_bn is None:
            last_layer_bn=bn

        super().__init__()
        ksz = _pair(kernel_size)
        self.bn = bn

        self.m = nn.Sequential(
            conv(in_channels,c1,kernel_size=ksz,stride=1,bn=bn,
                dropout=dropout, nonlinear=nonlinear, bias=bias),
            conv(c1,c1,kernel_size=ksz,stride=1,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c1,c1,kernel_size=ksz,stride=2,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c1,c2,kernel_size=ksz,stride=1,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c2,c2,kernel_size=ksz,stride=1,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c2,c2,kernel_size=ksz,stride=2,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c2,c2,kernel_size=ksz,stride=1,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c2,c2,kernel_size=ksz,stride=1,bn=bn, dropout=dropout,
                nonlinear=nonlinear, bias=bias),
            conv(c2,classes,kernel_size=(1,1),stride=1,bn=last_layer_bn, dropout=dropout,
                nonlinear=last_layer_nonlinear, bias=bias),
            Avg2d(),
            View(classes))



        self.nonlinear=nonlinear
        self.softmax=softmax

        #self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                w = p.data.view(p.size(0),-1)
                w.data.div_(w.norm(1,-1,keepdim=True))

    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])


    def forward(self, x):
        x = self.m(x)

        if self.softmax:
            x = x.softmax(dim=-1)

        return x
