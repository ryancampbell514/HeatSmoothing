'''CIFAR VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from .blocks import Linear, Conv
from torch.nn.modules.utils import _pair


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, bn=True, kernel_size=(3,3), 
            in_channels=3, classes=10,
            nonlinear=True, softmax=False, dropout=0., 
            last_layer_nonlinear=False, last_layer_bn=None, **kwargs ):
        super(VGG, self).__init__()

        conv =  Conv
        lin =  Linear
        kernel_size=_pair(kernel_size)


        def _make_layers(cfg, in_channels):
            layers = []
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [Conv(in_channels, x, kernel_size=kernel_size, padding=1,
                                    nonlinear=nonlinear, bn=bn, dropout=dropout)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)
        
        self.features = _make_layers(cfg[vgg_name], in_channels)
        cout = classes 
        self.classifier = Linear(512, cout, 
                nonlinear=last_layer_nonlinear, bn=last_layer_bn, 
                dropout=dropout)


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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.softmax:
            x = x.softmax(dim=-1)

        return x

def VGG11(**kwargs):
    return VGG('VGG11',False, **kwargs)

def VGG13(**kwargs):
    return VGG('VGG13',False, **kwargs)

def VGG16(**kwargs):
    return VGG('VGG16',False, **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19',False, **kwargs)
