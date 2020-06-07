""" MNIST example """

import os
import yaml
import ast, bisect
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import torchnet as tnt
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from flashlight import cvmodels as models
from flashlight.utils.loaders import get_model
from flashlight import dataloader
from flashlight.dataloader import cutout, optim_cutout
from flashlight.experiment_template.loss_functions import KL_loss
#from flashlight.cvmodels.mnist import LeNet

#from GaussianNets.utils import get_jacobian

parser = argparse.ArgumentParser('Gaussian Smoothing')
parser.add_argument('--info',type=str,default=None,metavar='INFO')
parser.add_argument('--std',type=float, default=0.1, metavar='DS',
        help = 'Standard Deviation of Gaussian')
parser.add_argument('--gamma',type=float, default=5, metavar='GA',
        help = 'Re-scaling hyperparameter')
parser.add_argument('--seed',type=float, default=0, metavar='S')
parser.add_argument('--batch-size',type=int, default=100, metavar='B')
parser.add_argument('--test-batch-size',type=int, default=1000, metavar='TB')
parser.add_argument('--cutout',type=int, default=0, metavar='N',
        help = 'Cutout size, if data loader supports cutout (default: 0)')
parser.add_argument('--add-gaussian',type=bool, default=False, metavar='G',
        help='Whether or not to add gaussian noise to the training data')
parser.add_argument('--dataset', type=str,help='dataset (default: "cifar10")',
        default='cifar10', metavar='DS',
        choices=['cifar10','cifar100', 'TinyImageNet','Fashion','mnist','svhn'])
parser.add_argument('--data-dir', type=str, required=False, default='/home/campus/oberman-lab/data', metavar='DIR',
        help='data storage directory')
parser.add_argument('--model', type=str, default='ResNet34', metavar='MOD')
parser.add_argument('--model-dir', type=str,
        default='/home/campus/ryan.campbell2/GaussianNets/cifar10/logs/models/base-model',
        metavar='DIR', help='The directory of the initial model.')
parser.add_argument('--model-args',type=str,
        default="{}",metavar='ARGS',
        help='A dictionary of extra arguments passed to the model.'
        ' (default: "{}")')
parser.add_argument('--bias',action='store_true', dest='bias',
        help = "Use model biases")
parser.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")
parser.add_argument('--no-bn',action='store_false', dest='bn',
       help = "Don't use batch norm")
parser.set_defaults(bn=True)
parser.add_argument('--kernel-size',type=int, default=3, metavar='K',
        help='convolution kernel size (default: 3)')
parser.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
parser.add_argument('--last-layer-nonlinear',
        action='store_true', default=False)
parser.add_argument('--classes',type=int, default=10,
        help='How many classes the model has')

parser.add_argument('--pth-name', type=str, default='best.pth.tar')
parser.add_argument('--parallel', action='store_true', dest='parallel',
        help='only allow exact matches to model keys during loading')
parser.add_argument('--strict', action='store_true', dest='strict',
        help='only allow exact matches to model keys during loading')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
parser.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
        default='[[0,1],[60,0.2],[120,0.04],[160,0.008]]')
parser.add_argument('--num-timesteps', type=int, default=5, metavar='NTS')
parser.add_argument('--num-epochs', type=int, default=200, metavar='NE')

args = parser.parse_args()

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

if args.seed is None:
    args.seed = int(time.time())
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA information
has_cuda = torch.cuda.is_available()
#has_cuda = False

# Get data
workers=4
test_loader = getattr(dataloader, args.dataset)(args.data_dir,
        mode='test', transform=False,
        batch_size=args.test_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=has_cuda)
image_shape = test_loader.image_shape
transforms = [cutout(args.cutout,channels=image_shape[0])]
train_loader = getattr(dataloader, args.dataset)(args.data_dir,
        mode='train', transform=True,
        batch_size=args.batch_size,
        training_transforms = transforms,
        num_workers=workers,
        shuffle=True,
        pin_memory=has_cuda,
        drop_last=True)

# Load in the original, baseline model, set it to training mode
classes = 10
model = get_model('./logs/models/base-model', classes, pth_name='best.pth.tar',
        parallel=args.parallel, strict=args.strict, has_cuda=has_cuda)
model.train()
for p in model.parameters():
    p.requires_grad_(True)
if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

# define the loss function (KL loss without reduction)
loss_function = KL_loss

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

def test():
    """ Evaluate current "model"'s argmax on test images """
    with torch.no_grad():
        model.eval()
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            if has_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.long()
            loss = loss_function(output, target)

            top1_train.add(output.data, target.view(-1).data)
            train_loss.add(loss.mean().item())

        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]

        test_loss = tnt.meter.AverageValueMeter()
        top1_test = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            if has_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.long()
            loss = loss_function(output, target)

            top1_test.add(output.data, target.view(-1).data)
            test_loss.add(loss.mean().item())

        t1 = top1_test.value()[0]
        l = test_loss.value()[0]

    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
            %(epoch, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
            %('Training loss',lt,t1t))

def train_more(ts,epoch):
    """ do the PDE-smoothing """

    model.train()
    u.eval()
    h = 1/args.num_timesteps  # step size
    for batch_ix, (x, y) in enumerate(train_loader):
        if has_cuda:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        Nb = len(x)
        xsh = x.shape

        x.requires_grad=True

        output = model(x)
        out_u = u(x)

        if epoch == 1 and batch_ix == 1:
            print('Model acc. %.3f'%((output.argmax(dim=-1) == y).float().sum()*100/args.batch_size))
            print('Init model acc. %.3f'%((out_u.argmax(dim=-1) == y).float().sum()*100/args.batch_size))

        u_lbls = out_u.argmax(dim=-1)
        Cu = out_u.argmax(dim=-1).view(-1)

        obj = (output -  out_u).norm(p=2,dim=-1).pow(2).div(2)

        #######################################################################

        # Johnson-Lindenstrauss and finite-differences to compute grad v
        penalty = torch.zeros(Nb)
        if has_cuda:
            penalty = penalty.cuda()
        num_reps = 10
        for j in range(num_reps):
            # compute model output - NEED TO FIX THIS
            # RIGHT NOW I AM DOING TOO MANY FORWARD PASSES
            x.requires_grad=True
            output = model(x)

            # compute some random vectors
            W = torch.randn_like(output,device=output.device) #* args.std
            W /= (classes**0.5)
            if has_cuda:
                W = W.cuda()

            wv_dot = (output * W).sum(dim=-1)

            # now do a finite-difference approximation for grad wv_dot
            grad_wv = grad(wv_dot.sum(), x, retain_graph=True)[0]
            sh = grad_wv.shape
            x = x.detach()
            v = grad_wv.view(sh[0],-1)
            nv = v.norm(2,dim=-1,keepdim=True)
            nz = nv.view(-1)>0
            v[nz] = v[nz].div(nv[nz])
            v = v.view(sh)
            dt = 0.1
            xf = x + dt*v  # forward Euler step

            xf.requires_grad=True
            outf = model(xf)

            forward = (outf * W).sum(dim=-1)
            backward = wv_dot

            pen = ((forward - backward) / dt).pow(2)
            penalty += pen

        #######################################################################

        objective = obj + args.gamma*0.5*h*(args.std**2)*penalty
        objective = objective.mean()

        objective.backward()
        optimizer.step()

        if batch_ix % 100 == 0 and batch_ix > 0:
            for param_group in optimizer.param_groups:
                learning = param_group['lr']
            print('[Timestep %2d, Epoch %2d, batch %3d, lr %.5f] dist. val %.5f, pen. val %.5g' %
                (ts, epoch, batch_ix, learning, obj.mean().item(), args.gamma*0.5*h*(args.std**2)*penalty.mean().item()))

if __name__=="__main__":

    # Now, iteratively solve heat equation to "smooth out" model
    num_timesteps = args.num_timesteps
    num_epochs = args.num_epochs
    for t in range(1, num_timesteps+1):

        # save model so far (call it "current"), load it back, call it 'u', set it to eval mode
        # at timestep 1, u will be the inital model
        args_file_path = os.path.join('./logs/models/current', 'args.yaml')
        with open(args_file_path, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        save_model_path = os.path.join('./logs/models/current', 'current.pth.tar')
        torch.save({'state_dict':model.state_dict()},save_model_path)

        u = get_model('./logs/models/current', classes, pth_name='current.pth.tar',
                      parallel=args.parallel, strict=args.strict, has_cuda=has_cuda)
        u.eval()
        for p in u.parameters():
            p.requires_grad_(False)
        if has_cuda:
            u = u.cuda()
            if torch.cuda.device_count()>1:
                u = nn.DataParallel(u)

        # Re-initialize "model" to random weights, this is the starting point of
        # the optimization algorithm (needed for each step in the discretized PDE solver)
        model_args = ast.literal_eval(args.model_args)
        model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                          kernel_size=args.kernel_size,
                          softmax=False,last_layer_nonlinear=args.last_layer_nonlinear,
                          dropout=args.dropout)
        model = getattr(models.cifar, args.model)(**model_args)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        if has_cuda:
            model = model.cuda()
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model)

        # define the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # reset lr scheduler (need to do this at the start of each timestep)
        def scheduler(optimizer,args):
            lS = np.array(ast.literal_eval(args.lr_schedule))
            #print(lS)
            llam = lambda e: float(lS[max(bisect.bisect_right(lS[:,0], e)-1,0),1])
            lscheduler = LambdaLR(optimizer, llam)
            return lscheduler
        schedule = scheduler(optimizer,args)

        # now minimize, sovle for v
        for epoch in range(1, num_epochs + 1):
            schedule.step()
            train_more(t,epoch)
            test()

    # save final model with arguments
    args_file_path = os.path.join('./logs/models/final', 'args.yaml')
    with open(args_file_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    save_model_path = os.path.join('./logs/models/final', 'final.pth.tar')
    torch.save({'state_dict':model.state_dict()},save_model_path)
