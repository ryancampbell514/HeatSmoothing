from torch import nn
from .resnet import ResNet
from .allcnn import AllCNNBase
from .blocks import Conv
from .vgg import VGG11, VGG13, VGG16, VGG19


def ResNet18(**kwargs):
    m = ResNet([2,2,2,2],**kwargs)
    return m

def ResNet22(**kwargs):
    m = ResNet([3,3,3],**kwargs)
    return m

def ResNet34(**kwargs):
    m = ResNet([5,5,5],**kwargs)
    return m

def ResNet50(**kwargs):
    m = ResNet([3,4,6,3],base_channels=64, block='Bottleneck',**kwargs)
    return m

def ResNet101(**kwargs):
    m = ResNet([3,4,23,3],base_channels=64, block='Bottleneck',**kwargs)
    return m

def ShakeShake34_2x32(**kwargs):
    m = ResNet([5,5,5],base_channels=32,
            block='BranchBlock',method='shake',**kwargs)
    return m

def ResNeXt34_2x32(**kwargs):
    m = ResNet([5,5,5],block='BranchBlock',
            base_channels=32,**kwargs)
    return m

def ResNeXt34_4x32(**kwargs):
    m = ResNet([5,5,5],block='BranchBlock',
            base_channels=32,branches=4,**kwargs)
    return m

class AllCNN(AllCNNBase):
    def __init__(self, **kwargs):
        super().__init__(Conv, **kwargs)
