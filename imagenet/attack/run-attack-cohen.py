import argparse, yaml
import os, sys

import numpy as np
import torch
from torch import nn

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append('../')

from HeatSmoothing.imagenet import resnet

# import attack
from salman_attacks import Attacker, PGD_L2, DDN

from HeatSmoothing.imagenet.cohen_utils.architectures import get_architecture

parser = argparse.ArgumentParser('Attack an IMAGENET-1K model with DDN or PGD'
                                  'Writes adversarial distances (and optionally images) to a npz file.')

groups0 = parser.add_argument_group('Required arguments')
groups0.add_argument('--datadir', type=str, default='/DIRECTORY/OF/IMAGENET/DATA',
        metavar='DIR', help='Directory where ImageNet data is saved')
groups0.add_argument('--model-path', type=str, required=True,metavar='PATH',
        help='Path to saved PyTorch model')
#groups0.add_argument('--pth-name', type=str, default='best.pth.tar')
groups0.add_argument('--parallel', action='store_true', dest='parallel',
        help='only allow exact matches to model keys during loading')
groups0.add_argument('--strict', action='store_true', dest='strict',
        help='only allow exact matches to model keys during loading')
groups0.add_argument('--criterion', type=str, default='top5',
        choices=['top5','cohen'], help='given a model and x, how to we estimate y?')
groups0.add_argument('--std', type=float, default=0.25,
        help='standard deviation of noise')
groups0.add_argument('--num-reps', type=int, default=25,
        help='number of Gaussian noise replication to do (when doing cohen-type stochastic prediction)')

groups2 = parser.add_argument_group('Optional attack arguments')
groups2.add_argument('--num-images', type=int, default=1000,metavar='N',
        help='total number of images to attack (default: 1000)')
groups2.add_argument('--batch-size', type=int, default=1,metavar='N',
        help='number of images to attack at a time (default: 100) ')
groups2.add_argument('--norm', type=str, default='L2',metavar='NORM',
        choices=['L2','Linf','L0','L1'],
        help='The dissimilarity metrics between images. (default: "L2")')
groups2.add_argument('--seed', type=int, default=0,
        help='seed for RNG (default: 0)')
groups2.add_argument('--random-subset', action='store_true',
        default=False, help='use random subset of test images (default: False)')

group1 = parser.add_argument_group('Attack hyperparameters')
group1.add_argument('--attack', default='DDN', type=str, choices=['DDN', 'PGD'])
group1.add_argument('--epsilon', default=4*256, type=float)  # want to force misclassification
group1.add_argument('--num-steps', default=20, type=int)
#group1.add_argument('--num-noise-vec', default=1, type=int,
#                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
group1.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")
# PGD-specific
group1.add_argument('--random-start', default=True, type=bool)
# DDN-specific
group1.add_argument('--init-norm-DDN', default=256.0, type=float)
group1.add_argument('--gamma-DDN', default=0.05, type=float)

args = parser.parse_args()

args.epsilon /= 256.0
args.init_norm_DDN /= 256.0
if args.criterion=='top5':
    args.num_reps = 1

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()

# Data loading code
if args.num_images<50000:
    IX = np.random.choice(50000, size=args.num_images, replace=False)
else:
    IX = np.arange(50000)
IX = torch.from_numpy(IX)

sampler = SubsetRandomSampler(IX)

valdir =os.path.join(args.datadir, 'validation/')
loader = torch.utils.data.DataLoader(
                    dataset = datasets.ImageFolder(valdir, transforms.Compose(
                            [transforms.Resize(int(288*1.14)),  # 256
                            transforms.CenterCrop(288),         # 224
                            transforms.ToTensor()])),
                    sampler=sampler,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

# retrieve pre-trained model
Nsamples=args.num_images
Nclasses=Nc=1000

checkpoint = torch.load(args.model_path)
model = get_architecture(checkpoint["arch"], 'imagenet')
model.load_state_dict(checkpoint['state_dict'])

model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if has_cuda:
    model = model.cuda()

# initialize the attack
if args.attack == 'PGD':
    print('Attacker is PGD.')
    attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
elif args.attack == 'DDN':
    print('Attacker is DDN.')
    attacker = DDN(steps=args.num_steps, device='cuda', max_norm=args.epsilon,
                init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
else:
    raise Exception('Unknown attack')

d0 = torch.full((args.num_images,),np.inf)
d2 = torch.full((args.num_images,),np.inf)
dinf = torch.full((args.num_images,),np.inf)
d1 = torch.full((args.num_images,),np.inf)

if has_cuda:
    d0 = d0.cuda()
    d2 = d2.cuda()
    dinf = dinf.cuda()
    d1 = d1.cuda()

K=0
for i, (x, y) in enumerate(loader):
    sys.stdout.write('Batch %2d/%d:\r'%(i+1,len(loader)))

    xsh = x.shape
    Nb = xsh[0]

    x = x.cuda()
    y = y.cuda()

    if args.criterion=='cohen':
        x = x.repeat((1,args.num_reps,1,1)).view(Nb*args.num_reps,*xsh[1:])
        noise = torch.randn_like(x,device='cuda') * args.std
    else:
        noise = None

    diff = attacker.attack(model, x, y,
                            noise=noise,
                            num_noise_vectors=args.num_reps,
                            no_grad=False
                            )
    #print(x.shape)
    #print(x_adv.shape)
    #exit()

    #diff = x_adv - x

    l0 = diff.view(Nb, -1).norm(p=0, dim=-1)
    l2 = diff.view(Nb, -1).norm(p=2, dim=-1)
    linf = diff.view(Nb, -1).norm(p=np.inf, dim=-1)
    l1 = diff.view(Nb,-1).norm(p=1,dim=-1)

    ix = torch.arange(K,K+Nb, device=x.device)

    d0[ix] = l0
    d2[ix] = l2
    dinf[ix] = linf
    d1[ix] = l1

    K+=Nb

md = d2.median()
me = d2.mean()
mx = d2.max()

print('\nDone. Statistics in norm: L2')
print('  Median adversarial distance: %.3g'%md)
print('  Mean adversarial distance:   %.3g'%me)
print('  Max adversarial distance:    %.3g'%mx)

if args.attack == 'PGD':
    st = 'l2-pgd'
    i = 0
    while os.path.exists('PGD/attack%s'%i):
        i +=1
    pth = os.path.join('PGD','attack%s/'%i)
elif args.attack == 'DDN':
    st = 'l2-ddn'
    i = 0
    while os.path.exists('DDN/attack%s'%i):
        i +=1
    pth = os.path.join('DDN','attack%s/'%i)

os.makedirs(pth, exist_ok=True)

args_file_path = os.path.join(pth, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

dists = {'index':IX.cpu().numpy(),
         'l0':d0.cpu().numpy(),
         'l2':d2.cpu().numpy(),
         'linf':dinf.cpu().numpy(),
         'l1': d1.cpu().numpy()}

with open(os.path.join(pth, st+'.npz'), 'wb') as f:
    np.savez(f, **dists)

