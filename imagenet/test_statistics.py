""" Load in a trained model, calculate the L-bound as well as top1/top5 accuracies """

import argparse
import os, sys

import numpy as np
import pandas as pd
import math
import pickle as pk
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
sys.path.append('../')
import resnet

# FOR A COHEN MODEL, UNCOMMENT THE NEXT LINE
#from HeatSmoothing.imagenet.cohen_utils.architectures import get_architecture

# FOR A SALMAN MODEL, UNCOMMENT THE NEXT LINE
#from HeatSmoothing.imagenet.cohen_utils.architectures import get_architecture

parser = argparse.ArgumentParser('Gathers statistics of a model on the test'
        'set, and saves these statistics to a pickle file in the model directory')

parser.add_argument('--datadir', type=str, default='/PATH/TO/IMAGENET/DATASET/DIRECTORIES',
        metavar='DIR', help='Directory where ImageNet data is saved')
parser.add_argument('--model-path', type=str, required=True,metavar='PATH',
        help='Path to saved PyTorch model')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='resnet50',
        choices=['smoothresnet50','resnet50'], help='Model')
parser.add_argument('--num-images', type=int, default=50000,metavar='N',
        help='total number of images to attack (default: 50000)')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time')
parser.add_argument('--norm', default='L2', choices=['L2', 'Linf'],
        help='norm measuring adversarial perturbations')
parser.add_argument('--std', type=float, default=0.25,
        help='standard deviation of added gaussian noise')
parser.add_argument('--num-reps', type=int, default=25)
parser.add_argument('--is-cohen',action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()
#has_cuda = False

# Data  and model loading code
# ----------------------------
if args.num_images<50000:
    IX = np.random.choice(50000, size=args.num_images, replace=False)
else:
    IX = np.arange(50000)
IX = torch.from_numpy(IX)

sampler = SubsetRandomSampler(IX)

valdir =os.path.join(args.datadir, 'validation/')
loader = torch.utils.data.DataLoader(
                    dataset = datasets.ImageFolder(valdir, transforms.Compose(
                            [transforms.Resize(int(288*1.14)), # 256  # 288*1.14
                            transforms.CenterCrop(288),        # 224  # 288
                            transforms.ToTensor()])),
                    sampler=sampler,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

Nsamples=args.num_images
Nclasses=Nc=1000

# FOR A BASELINE MODEL, USE THE FOLLOWING 8 LINES TO LOAD IN THE PRETRAINED MODEL
model = getattr(resnet, args.model)().cuda()
savedict = torch.load(args.model_path,map_location='cpu')
model.load_state_dict(savedict['state_dict'])
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
model = nn.Sequential(nn.BatchNorm2d(3,affine=False), model)
model[0].running_var =  torch.tensor(std)**2
model[0].running_mean = torch.tensor(mean)

# FOR A COHEN OR SALMAN MODEL, USE THE FOLLOWING 3 LINES TO LOAD IN THE PRETRAINED MODEL
#checkpoint = torch.load(args.model_path)
#model = get_architecture(checkpoint["arch"], 'imagenet')
#model.load_state_dict(checkpoint['state_dict'])

model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)


# max_{i\not \in top5} p_i - p_c
def criterion(z,y):
    #p = z.softmax(dim=-1)
    ix = torch.arange(z.shape[0],device=z.device)
    #pc = p.clone()
    #pc[ix,y] = 0.
    ps = p.sort(dim=-1,descending=True)[0]

    return (ps[:,5:]).max(dim=-1)[0] - p[ix,y]

Loss = torch.zeros(Nsamples).cuda()
NormGradLoss = torch.zeros(Nsamples).cuda()
Top1 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()
Rank = torch.zeros(Nsamples,dtype=torch.int64).cuda()
Top5 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()

sys.stdout.write('\nRunning through dataloader:\n')
Jx = torch.arange(Nc).cuda().view(1,-1)
Jx = Jx.expand(args.batch_size, Nc)
for i, (x,y) in enumerate(loader):
    sys.stdout.write('  Completed [%6.2f%%]\r'%(100*i*args.batch_size/Nsamples))
    sys.stdout.flush()

    xsh = x.shape
    Nb = xsh[0]

    if has_cuda:
        x, y = x.cuda(), y.cuda()
    #x = x + (torch.randn_like(x,device=x.device) * args.std)

    x.requires_grad_(True)

    if args.is_cohen:
        x = x.unsqueeze(1).repeat(1,args.num_reps,1,1,1)
        noise = torch.randn_like(x).cuda() * args.std

        xn = x + noise
        xn = xn.view(Nb*args.num_reps,*xsh[1:])

        yhat = model(xn)
        p = yhat.softmax(dim=-1).view(Nb,args.num_reps,-1)
        p = p.mean(dim=1)

    else:
        yhat = model(x)
        p = yhat.softmax(dim=-1)

    psort , jsort = p.sort(dim=-1,descending=True)
    b = jsort==y.view(-1,1)
    rank = Jx[b]
    pmax = psort[:,0]
    logpmax = pmax.log()

    p5,ix5 = psort[:,0:5], jsort[:,0:5]
    ix1 =  jsort[:,0]
    sump5 = p5.sum(dim=-1)

    loss = criterion(p, y)
    g = grad(loss.sum(),x)[0]
    if args.norm=='L2':
        gn = g.reshape(len(y),-1).norm(dim=-1)
    elif args.norm=='Linf':
        gn = g.view(len(y),-1).norm(p=1,dim=-1)

    top1 = torch.tensor((ix1==y).clone().detach(), dtype=torch.uint8)
    top5 = (ix5==y.view(args.batch_size,1)).sum(dim=-1)

    ix = torch.arange(i*args.batch_size, (i+1)*args.batch_size,device=x.device)

    Loss[ix] = loss.detach()
    Rank[ix]= rank.detach()
    Top1[ix] = top1.detach().cuda()
    Top5[ix] = top5.detach().type(torch.uint8)
    NormGradLoss[ix] = gn.detach()
sys.stdout.write('   Completed [%6.2f%%]\r'%(100.))

df = pd.DataFrame({'loss':Loss.cpu().numpy(),
                   'top1':np.array(Top1.cpu().numpy(),dtype=np.bool),
                   'top5':np.array(Top5.cpu().numpy(), dtype=np.bool),
                   'norm_grad_loss':NormGradLoss.cpu().numpy(),
                   'rank': Rank.cpu().numpy()})

print('\n\ntop1 error: %.2f%%,\ttop5 error: %.2f%%'%(100-df['top1'].sum()/Nsamples*100, 100-df['top5'].sum()/Nsamples*100))

Lmax = NormGradLoss.max()
Lmean = NormGradLoss.mean()
dualnorm = 'L1' if args.norm=='Linf' else 'L2'
print('mean & max gradient norm (%s): %.2f, %.2f'%(dualnorm, Lmean, Lmax))

LossGap = (-Loss).clamp(0)
denom = (1/args.std) * ((2/math.pi)**0.5)
Lbound = LossGap/denom
#Lbound = LossGap/Lmax
df['Lbound'] = Lbound.cpu().numpy()
print('mean 1st order lower bound on adversarial distance (%s): %.2g'%(args.norm, Lbound[Lbound>0].mean()))
print('                                                 median: %.2g'%(Lbound[Lbound>0].median()))

ix1 = np.array(df['top1'], dtype=bool)
ix5 = np.array(df['top5'], dtype=bool)
ix15 = np.logical_or(ix5,ix1)
ixw = np.logical_not(np.logical_or(ix1, ix5))

df['type'] = pd.DataFrame(ix1.astype(np.int8) + ix5.astype(np.int8))
d = {0:'mis-classified',1:'top5',2:'top1'}
df['type'] = df['type'].map(d)
df['type'] = df['type'].astype('category')
df['ix'] = IX.numpy() 

basename = args.model_path.split('.pth.tar')
pklpath = basename[0]+'-stats-%s.pkl'%args.norm
df.to_pickle(pklpath)
