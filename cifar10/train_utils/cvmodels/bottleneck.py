import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math
from math import inf

from .blocks import Conv

class Bottleneck(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear=True, 
            dropout=0., **kwargs):
        """A basic 2d ResNet bottleneck block, with modifications on original ResNet paper
        [1].  Every convolution is followed by batch normalization (if active).

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.nonlinear=True
        self.bn = bn
        self.register_buffer('h', th.tensor(1.))

        self.conv0 = Conv(channels, channels//4, kernel_size=(1,1), bn=bn, dropout=dropout,
                nonlinear=nonlinear)
        self.conv1 = Conv(channels//4, channels//4, kernel_size=kernel_size, bn=bn, dropout=dropout,
                nonlinear=nonlinear)
        self.conv2 = Conv(channels//4, channels, kernel_size=(1,1), bn=bn, dropout=dropout, 
                nonlinear=False)


    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)

        y = (x+y)/self.h

        if self.nonlinear:
            y = F.relu(y)

        return y


    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'
             
        return s.format(**self.__dict__)
