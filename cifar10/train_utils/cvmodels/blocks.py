import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math
from math import inf, sqrt, pi


from .shake import shake

def smoothrelu(x,s=1.):
    return 0.5 * (x * th.erf(s*x/sqrt(2)) 
                + sqrt(2/pi) / s * th.exp(-(s*x)**2 / 2)
                + x )


def prod(x):
    p = 1
    for x_ in x:
        p *=x
    return p


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim,  bn=True, nonlinear='relu', dropout=0.,
            bias=True, **kwargs):
        """A linear block.  The linear layer is followed by batch
        normalization (if active) and a ReLU (again, if active)

        Args:
            in_dim: number of input dimensions
            out_dim: number of output dimensions
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(th.randn(out_dim)) 
        else:
            self.register_parameter('bias', None)
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.nonlinear=nonlinear
        if bn:
            self.bn = nn.BatchNorm1d(out_dim, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.reset_parameters()


    def reset_parameters(self):
        n = self.in_dim
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.linear(x, self.weight, None)

        if self.bn:
            y = self.bn(y)

        if self.bias is not None:
            b = self.bias.view(1,-1)
            y = y+b

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        return y

    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1, padding=None,
            kernel_size=(3,3),  bn=True, nonlinear='relu', dropout = 0.,
            bias=True,
            **kwargs):
        """A 2d convolution block.  The convolution is followed by batch
        normalization (if active).

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(th.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size=_pair(kernel_size)
        self.nonlinear=nonlinear
        if padding is None:
            self.padding = tuple([k//2 for k in kernel_size])
        else:
            self.padding = _pair(padding)

        # this is where bn is defined
        if bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.conv2d(x, self.weight, None, self.stride, self.padding,
                1, 1)

        if self.bn:
            y = self.bn(y)

        if self.bias is not None:
            b = self.bias.view(1,self.out_channels,1,1)
            y = y+b

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        return y

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'
        return s.format(**self.__dict__)



class BasicBlock(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear='relu',
            dropout=0., **kwargs):
        """A basic 2d ResNet block, with modifications on original ResNet paper
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
        self.nonlinear='relu'
        self.bn = bn

        self.conv0 = Conv(channels, channels, kernel_size=kernel_size, bn=bn, dropout=dropout,
                nonlinear=nonlinear)
        self.conv1 = Conv(channels, channels, kernel_size=kernel_size, bn=bn, dropout=dropout,
                nonlinear=False)


    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)

        y = x+y

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        return y

    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class Bottleneck(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear='relu',
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
        self.nonlinear=nonlinear
        self.bn = bn

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

        y = x+y

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        return y


    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class BranchBlock(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, branches=2,
            method='mean', nonlinear='relu', dropout = 0., **kwargs):
        """A 2d ResNet block, where the channels are 'branched' into separate
        groups.  Every convolution is followed by batch normalization (if active).

        The branches are aggregated either by taking the mean, max, or min,
        depending on the argument "method". Default is to take the mean.  If
        method='shake', then a mean is taken during model evaluation, but a
        random convex combination is used during training.

        When method is 'mean', this module is like a simplified ResNeXt block,
        without the 1x1 convolution layer aggregating the channels, using a
        mean to aggregate instead.

        When method is 'max', this module is a MaxOut ResNet.

        When method is 'shake', concentration is 1, and branches=2, this is
        a Shake-Shake Net.

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)
            branches (int, optional): number of branches (default: 2)
            method (string, optional): aggregation method. One of 'mean',
                'max', 'min' or 'shake'. (default: mean)
            concentration (float, optional): if method='shake', this is the
                concentration hyperparameter of the Dirichlet distribution
                generating the convex combination (default:1)
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.branches = branches
        self.method = method

        if method=='shake':
            try:
                self.concentration = kwargs['concentration']
            except:
                self.concentration = 1.
        else:
            self.concentration = None

        assert method in ['shake', 'mean', 'min', 'max']

        padding = [k //2 for k in kernel_size]

        self.conv0 = nn.Conv2d(channels, channels*branches,
                                kernel_size, groups=1,
                                bias=True, padding=padding, stride=1)
        self.conv1 = nn.Conv2d(channels*branches, channels*branches,
                                kernel_size, groups=branches,
                                bias=True, padding=padding, stride=1)

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(channels * branches, affine=False)
            self.bn1 = nn.BatchNorm2d(channels * branches, affine=False)

        self.nonlinear = nonlinear


    def forward(self, x):

        if self.dropout:
            y = self.dropout(x)
        else:
            y = x

        y = self.conv0(y)

        if self.bn:
            y = self.bn0(y)

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        if self.dropout:
            y = self.dropout(y)

        y = self.conv1(y)

        if self.bn:
            y = self.bn1(y)

        y = y.unsqueeze(1).chunk(self.branches,dim=2)
        y = th.cat(y, 1)

        if self.method=='shake':
            if self.nonlinear:
                y = shake(y, 1, 0, self.concentration, self.training)
            else:
                y = y.mean(1)
        elif self.method=='mean':
            y = y.mean(1)
        elif self.method=='max':
            if not self.nonlinear:
                raise ValueError('Linear evaluation not possible with aggregation method "max"')
            y = y.max(1)[0]
        elif self.method=='min':
            if not self.nonlinear:
                raise ValueError('Linear evaluation not possible with aggregation method "min"')
            y = y.min(1)[0]

        y = x+y
        
        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        return y

    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}, method={method}')

        if self.method=='shake':
            s += ', concentration={concentration}'

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)
