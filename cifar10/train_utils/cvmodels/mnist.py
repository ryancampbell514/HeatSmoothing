import torch.nn as nn
import torch as th
import torch.nn.functional as F

from .utils import View
from .blocks import Conv, Linear


class LeNet(nn.Module):
    def __init__(self, bn=True, classes=10, dropout=0., bias=True,
            softmax=False, last_layer_nonlinear=False, **kwargs):
        """Implementation of LeNet [1].

        [1] LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to
               document recognition. Proceedings of the IEEE. 1998 Nov;86(11):2278-324."""
        super().__init__()

        def conv(ci,co,ksz,psz,dropout=0,bn=True):
            conv_ = Conv(ci,co,kernel_size=ksz, padding=0, bn=bn, bias=bias)

            m = nn.Sequential(
                conv_,
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(dropout))
            return m

        self.m = nn.Sequential(
            conv(1,20,(5,5),3,dropout=dropout, bn=bn),
            conv(20,50,(5,5),2,dropout=dropout, bn=bn),
            View(200),
            Linear(200, 500, bn=bn, bias=bias),
            nn.Dropout(dropout),
            Linear(500,classes, bn=bn, last_layer_nonlinear=last_layer_nonlinear,
                bias=bias))

        self.softmax=softmax
        self.bn = bn

        #for n, p in self.named_parameters():
        #    if 'weight' in n:
        #        w = p.data.view(p.size(0),-1)
        #        w.div_(w.norm(1,-1,keepdim=True))

    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])


    def forward(self, x):
        x = self.m(x)

        if self.softmax:
            x = x.softmax(dim=-1)
        
        return x

class LinearModel(th.nn.Module):
    def __init__(self, D_in=28*28, H=100, D_out=10):
        super(LinearModel,self).__init__()
        self.linear1 = th.nn.Linear(D_in, H)
        self.linear2 = th.nn.Linear(H, D_out)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        y_pred = self.linear2(x)
        return y_pred
