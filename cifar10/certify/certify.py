""" import a model, check the cohen bound """

import time
import yaml
import os
import pandas as pd
import torch
import torchvision
import numpy as np
import sys
from statistics import mean
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import Subset
from torch.autograd import Variable, grad
import torchnet as tnt
import torch.nn.functional as F

#from robustness.datasets import CIFAR
#from robustness.model_utils import make_and_restore_model

from HeatSmoothing.cifar10.train_utils.loaders import get_model
#from flashlight.utils import cohen
#from flashlight.criteria import top1
from HeatSmoothing.cifar10.train_utils import dataloader
#from flashlight.experiment_template.loss_functions import KL_loss, cw_loss

from HeatSmoothing import cohen

import argparse
parser = argparse.ArgumentParser('Certify a model using the method from the Cohen Randomized Smoothing paper')

parser.add_argument('--data-dir', type=str, default='/home/campus/oberman-lab/data/',
        help='data storage directory')
parser.add_argument('--dataset', type=str,help='dataset (default: "cifar10")',
        default='cifar10', metavar='DS',
        choices=['cifar10','cifar100', 'TinyImageNet','Fashion','mnist','svhn'])
parser.add_argument('--model-dir', type=str, default=None, metavar='DIR',
        help='for loading a trained model')
parser.add_argument('--std', type=float, default=0.1, metavar='SD',
        help = 'standard deviation of the added gaussian noise')
parser.add_argument('--seed', type=int, default=0, metavar='S',
        help='random seed')
parser.add_argument('--pth-name', type=str, default='best.pth.tar')
parser.add_argument('--parallel', action='store_true', dest='parallel',
        help='only allow exact matches to model keys during loading')
parser.add_argument('--strict', action='store_true',
        help='only allow exact matches to model keys during loading')
parser.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")

parser.add_argument('--num-images',type=int,default=1000)
parser.add_argument('--is-cohen',action='store_true')
args = parser.parse_args()


print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

if args.seed is None:
    args.seed = 0
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA information
has_cuda = torch.cuda.is_available()
#has_cuda = False

# import our model
classes = 10
criterion = top1
model = get_model(args.model_dir, classes, pth_name=args.pth_name,
        parallel=args.parallel, strict=args.strict, has_cuda=has_cuda)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

# import data
transform = transforms.Compose([transforms.ToTensor()])
classes = Nc = 10
root = os.path.join(args.data_dir,'cifar10')
ds = CIFAR10(root, download=True, train=False, transform=transform)
ix = torch.arange(args.num_images)
subset = Subset(ds, ix)
loader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=1, shuffle=False,
                    num_workers=4, pin_memory=True)

label = []
predict = []
radius = []
correct = []
times = []
ctimes = []

for idx, (x,y) in enumerate(loader):
    sys.stdout.write('[image: %d / %d]\r'%(idx,args.num_images))

    x = x[0]
    xsh = x.shape
    if has_cuda:
        x,y = x.cuda(),y.cuda()

    label.append(y.cpu().item())

    start_time = time.time()
    cohen_pred_class, cohen_radius, class_time = cohen.certify(model, x, std=args.std, is_cohen=args.is_cohen, n=10000, rule='top1', classes=10)
    end_time = time.time()
    predict.append(cohen_pred_class)
    correct.append((cohen_pred_class == y).item())
    radius.append(cohen_radius)
    times.append(end_time - start_time)
    ctimes.append(class_time)

df = {'label':np.array(label),
      'predict':np.array(predict),
      'radius':np.array(radius),
      'correct':np.array(correct),
      'cert time':np.array(times),
      'class time':np.array(ctimes),
      }

df = pd.DataFrame(df)
print(df)

print('Avg. classification time:', mean(ctimes))
print('Avg. certification time:', mean(times))

# save data along with arguments
i = 0
pth = os.path.join('data%s/'%i)
while os.path.exists(pth):
    pth = os.path.join('data%s/'%i)
    i +=1
os.makedirs(pth, exist_ok=True)
args_file_path = os.path.join(pth, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)
with open(os.path.join(pth, 'data.pkl'), 'wb') as f:
    df.to_pickle(f)
