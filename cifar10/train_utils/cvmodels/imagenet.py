from torch import nn
from .resnet import ResNet

def ResNet18(**kwargs):
    m = ResNet([2,2,2,2],conv0_kwargs={'kernel_size':(7,7), 'stride':2,'padding':3},
            base_channels=64, conv0_pool=nn.MaxPool2d(kernel_size=3,stride=2),
            conv0_maxpool=True,**kwargs)
    return m

def ResNet34(**kwargs):
    m = ResNet([3,4,6,3],conv0_kwargs={'kernel_size':(7,7), 'stride':2,'padding':3},
            base_channels=64,
            conv0_pool=nn.MaxPool2d(kernel_size=3,stride=2), **kwargs)
    return m

def ResNeXt34_4x32(**kwargs):
    m = ResNet([3,4,6,3],block='BranchBlock', 
            conv0_kwargs={'kernel_size':(7,7), 'stride':2,'padding':3},
            base_channels=32,
            conv0_pool=nn.MaxPool2d(kernel_size=3,stride=2), branches=4,
            **kwargs)
    return m

