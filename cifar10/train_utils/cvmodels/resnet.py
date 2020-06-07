import torch as th
from torch import nn
from torch.nn.modules.utils import _pair
from . import blocks

from .utils import View, Avg2d
from .blocks import Conv, Linear


class ResNet(nn.Module):

    def __init__(self, layers, block='BasicBlock', in_channels=3,
                 classes=10, kernel_size=(3,3),  nonlinear='relu',
                 conv0_kwargs = {'kernel_size':(3,3), 'stride':1},
                 conv0_pool=None, downsample_pool=nn.AvgPool2d, dropout = 0.,
                 last_layer_nonlinear=False, last_layer_bn=None,
                 softmax=False, bn=True, base_channels=16, **kwargs):
        if last_layer_bn is None:
            last_layer_bn=bn

        super().__init__()
        kernel_size = _pair(kernel_size)

        def make_layer(n, block, in_channels, out_channels, stride):
            sublayers = []
            if not in_channels==out_channels:
                sublayers.append(Conv(in_channels, out_channels, kernel_size=(1,1), dropout=dropout,
                    nonlinear=nonlinear, bn=bn))

            if stride>1:
                sublayers.append(downsample_pool(stride))

            for k in range(n):
                sublayers.append(block(out_channels, kernel_size=kernel_size,
                    bn=bn, nonlinear=nonlinear, dropout = dropout, **kwargs))

            return nn.Sequential(*sublayers)


        block = getattr(blocks, block)

        self.layer0 = Conv(in_channels, base_channels, **conv0_kwargs,
                bn=bn, nonlinear=nonlinear, dropout=dropout)

        if conv0_pool:
            self.maxpool = conv0_pool
        else:
            self.maxpool = False


        _layers = []
        for i, n in enumerate(layers):

            if i==0:
                _layers.append(make_layer(n, block, base_channels,
                    base_channels, 1))
            else:
                _layers.append(make_layer(n, block, base_channels*(2**(i-1)),
                    base_channels*(2**i), 2))

        self.layers = nn.Sequential(*_layers)

        self.pool = Avg2d()
        self.view = View((2**i)*base_channels)

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.fc = Linear((2**i)*base_channels, classes, nonlinear=last_layer_nonlinear, bn=last_layer_bn)

        self.nonlinear=nonlinear
        self.softmax = softmax
        self.bn = bn  # this is where batch norm is executed, a boolean

    def fuse_bn(self):
        if self.bn:
            self.fc.fuse_bn()
            self.layer0.fuse_bn()
            for L in self.layers:
                for l in L:
                    try:
                        l.fuse_bn()
                    except AttributeError as e:
                        pass
            self.bn=False

    def normalize(self):
        self.fc.normalize()
        self.layer0.normalize(norm='2,inf')
        for L in self.layers:
            for l in L:
                try:
                    l.normalize()
                except AttributeError as e:
                    pass

    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])


    def forward(self, x):
        x = self.layer0(x)
        if self.maxpool:
            x = self.maxpool(x)
        x = self.layers(x)
        x = self.pool(x)
        x = self.view(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)

        if self.softmax:
            x = x.softmax(dim=-1)

        return x


