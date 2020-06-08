""" import a model, check the cohen bound (top-1)"""

import time
import yaml
import os
import pandas as pd
import torch
import torchvision
import numpy as np
import sys
import torch.nn as nn
from statistics import mean
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import Subset
from torch.autograd import Variable, grad
import torchnet as tnt
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler

#from flashlight.utils.loaders import get_model
#from flashlight.criteria import top1
#from flashlight import dataloader
#from flashlight.experiment_template.loss_functions import KL_loss, cw_loss

from HeatSmoothing import cohen
from HeatSmoothing.imagenet import resnet

import argparse
parser = argparse.ArgumentParser('Certify a model')

parser.add_argument('--datadir', type=str, default='/mnt/data/scratch/data/imagenet',
        help='data storage directory')
parser.add_argument('--model-path', type=str, default=None, metavar='DIR',
        help='for loading a trained model')
parser.add_argument('--model', type=str, default='resnet50',
        choices=['smoothresnet50','resnet50'], help='Model')
parser.add_argument('--std', type=float, default=None, required=True, metavar='SD',
        help = 'standard deviation of the added gaussian noise')
parser.add_argument('--seed', type=int, default=0, metavar='S',
        help='random seed')

parser.add_argument('--num-images',type=int,default=1000,
        help='numer of images to certify (default=1000,max=5000)')
parser.add_argument('--is-cohen',action='store_true')
parser.add_argument('--rule', type=str, default='top1', metavar='R',
        choices=['top1', 'top5'], help='classification criterion')
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

# import data (only load in one image at a time)
if args.num_images<50000:
    IX = np.random.choice(50000, size=args.num_images, replace=False)
else:
    IX = np.arange(50000)
IX = torch.from_numpy(IX)
sampler = SubsetRandomSampler(IX)
valdir =os.path.join(args.datadir, 'validation/')
loader = torch.utils.data.DataLoader(
                    dataset = datasets.ImageFolder(valdir, transforms.Compose(
                            [transforms.Resize(int(288*1.14)),
                            transforms.CenterCrop(288),
                            transforms.ToTensor()])),
                    sampler=sampler,
                    batch_size=1, shuffle=False,
                    num_workers=4, pin_memory=True)

# import model
model = getattr(resnet, args.model)()
Nsamples=args.num_images
Nclasses=Nc=1000
savedict = torch.load(args.model_path,map_location='cpu')
model.load_state_dict(savedict['state_dict'])
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
model = nn.Sequential(nn.BatchNorm2d(3,affine=False), model)
model[0].running_var =  torch.tensor(std)**2
model[0].running_mean = torch.tensor(mean)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

# store data in these lists:
label = []        # true class
predict = []      # predicted class
radius = []       # L2 radius
correct = []      # predicted class == true class
class_times = []  # time it takes to classify
cert_times = []   # time it takes to certify

for idx, (x,y) in enumerate(loader):
    sys.stdout.write('[image: %d / %d]\r'%(idx,args.num_images))

    x = x[0]
    xsh = x.shape
    if has_cuda:
        x,y = x.cuda(),y.cuda()

    label.append(y.cpu().item())

    start_time = time.time()
    cohen_pred_class, cohen_radius, class_time = cohen.certify(model, x, std=args.std, is_cohen=args.is_cohen, n0=25, n=1000, classes=Nc, rule=args.rule)
    end_time = time.time()
    if args.rule=='top1':
        predict.append(cohen_pred_class)
        correct.append((cohen_pred_class == y).item())
    elif args.rule=='top5':
        correct.append(int(y in cohen_pred_class))  # equals 1 if true class is in predicted top-5
    radius.append(cohen_radius)
    class_times.append(class_time)
    cert_times.append(end_time - start_time)

if args.rule=='top1':
    df = {'label':np.array(label),
          'predict':np.array(predict),
          'radius':np.array(radius),
          'correct':np.array(correct),
          'class time':np.array(class_times),
          'cert time':np.array(cert_times)
          }
elif args.rule=='top5':
    df = {'label':np.array(label),
          'radius':np.array(radius),
          'correct':np.array(correct),
          'class time':np.array(class_times),
          'cert time':np.array(cert_times)
          }

# print dataframe
df = pd.DataFrame(df)
print(df)

print('Avg. classification time:', mean(class_times))
print('Avg. certification time:', mean(cert_times))

# save data along with arguments
i = 0
pth = os.path.join('./certify','data%s/'%i)
while os.path.exists(pth):
    pth = os.path.join('./certify','data%s/'%i)
    i +=1
os.makedirs(pth, exist_ok=True)
args_file_path = os.path.join(pth, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)
with open(os.path.join(pth, 'data.pkl'), 'wb') as f:
    df.to_pickle(f)
