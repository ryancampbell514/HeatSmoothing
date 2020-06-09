import yaml
import argparse
import collections
import copy
import gc
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
import numpy as np

import torch.backends.cudnn as cudnn
from torch import nn
#import torch.distributed as dist
import torch.optim
import torch.utils.data
#import torch.utils.data.distributed
from torch.autograd import grad

import dataloader
#import dist_utils
import experimental_utils
import resnet
from logger import TensorboardLogger, FileLogger
from meter import AverageMeter, NetworkMeter, TimeMeter

parser = argparse.ArgumentParser(description='Train a ResNet50 on ImageNet using our deterministic approach.')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--phases', type=str,
                    help='Specify epoch order of data resize and learning rate schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--init-bn0', action='store_true',
                    help='Facebook batch norm hack')
parser.add_argument('--nonlinearity', default='relu', type=str, choices=['relu','c2relu'],
                    help='type of nonlinearity')
parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N',
                    help='log/print every this many steps (default: 5)')
parser.add_argument('--logdir', default=None, type=str,
                    help='where logs go')
parser.add_argument('--std', type=float, default=0.25,
                    help='std of Normal distribution')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='scaling correction term')
parser.add_argument('--num-timesteps', type=int, default=5,
                    help='number of finite-difference timesteps to perform')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train-batch-size', type=int, default=32)
parser.add_argument('--val-batch-size', type=int, default=100)
#parser.add_argument('--distributed', action='store_true',
#                    help='Run distributed training. Default True')
#parser.add_argument('--dist-url', default='env://', type=str,
#                    help='url used to set up distributed training')
#parser.add_argument('--dist-backend', default='nccl', type=str,
#                    help='distributed backend')
#parser.add_argument('--local_rank', default=0, type=int,
#                    help='Used for multi-process training. Can either be manually set or automatically set by using \'python -m multiproc\'.')
parser.add_argument('--start-epoch', type=int, default=0,
                    help='for debugging purposes.')
parser.add_argument('--end-epoch', type=int, default=None)

parser.add_argument('--init-pth', type=str, required=True,
                    help='path to initial model f^0 (.pth.tar file)')

cudnn.benchmark = True
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#is_master = (not args.distributed) or (dist_utils.env_rank()==0)
#is_rank0 = args.local_rank == 0
tb = TensorboardLogger(args.logdir, is_master=True)
log = FileLogger(args.logdir, is_master=True, is_rank0=True)

# KL loss (for now)
kl_div = nn.KLDivLoss(reduction='none').cuda()

def main():
    # print logs
    log.console(args)
    #tb.log('sizes/world', dist_utils.env_world_size())

    #if args.distributed:
    #    log.console('Distributed initializing process group')
    #    torch.cuda.set_device(args.local_rank)
    #    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=dist_utils.env_world_size())
    #    assert(dist_utils.env_world_size() == dist.get_world_size())
    #    log.console("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))

    # import the initial model
    model = getattr(resnet, 'resnet50')(bn0=args.init_bn0, nonlinearity=args.nonlinearity).cuda()
    #if args.distributed:
    #    model = dist_utils.DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    #init_pth = '/home/campus/christopher.finlay/other-repos/imagenet18/training/runs/run-19-04-21T170801/model_best.pth.tar'
    #init_pth = './runs/base-model/model_best.pth.tar'
    #init_pth = './runs/model-TS2.pth.tar'
    init_pth = args.init_pth
    savedict = torch.load(init_pth,map_location='cpu')
    model.load_state_dict(savedict['state_dict'])
    for p in model.parameters():
        p.requires_grad_(True)

    model = nn.DataParallel(model.cuda()).cuda()

    # now iterate through the timesteps, performing SGD optimization at each timestep
    for ts in range(1,args.num_timesteps+1):

        # 1- define our current 'frozen' model (call it 'curr_mod'), and set its weights to those of 'model'
        curr_mod = getattr(resnet, 'resnet50')(bn0=args.init_bn0, nonlinearity=args.nonlinearity).cuda()
        #if args.distributed:
        #    curr_mod = dist_utils.DDP(curr_mod, device_ids=[args.local_rank], output_device=args.local_rank)
        curr_mod.load_state_dict(model.module.state_dict())
        curr_mod = nn.DataParallel(curr_mod.cuda()).cuda()
        curr_mod.eval()
        for p in curr_mod.parameters():
            p.requires_grad_(False)


        # 2- reinitialize 'model' to random initial weights
        model = resnet.resnet50(bn0=args.init_bn0, nonlinearity=args.nonlinearity).cuda()
        #if args.distributed:
            #model = dist_utils.DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        model = nn.DataParallel(model.cuda()).cuda()
        global model_params, master_params
        model_params = master_params = model.parameters()
        bparams, oparams = [], []
        for name, param in model.named_parameters():
            if 'bias' in name:
                bparams.append(param)
            else:
                oparams.append(param)
        optim_params = [{'params':bparams, 'weight_decay':0.},
                        {'params':oparams, 'weight_decay':args.weight_decay}]

        # 3- define SGD optimizer
        optimizer = torch.optim.SGD(params=optim_params, lr=0, momentum=args.momentum, weight_decay=args.weight_decay)

        # 4- import dataloader and set the learning rate scheduler
        log.console("Creating data loaders")
        phases = eval(args.phases)
        dm = DataManager([copy.deepcopy(p) for p in phases if 'bs' in p])
        scheduler = Scheduler(optimizer, [copy.deepcopy(p) for p in phases if 'lr' in p])

        # 5- Go through data
        #if args.distributed:
        #    log.console('Syncing machines before training')
        #    dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

        log.event("~~epoch\thours\ttop1\ttop5\n")

        if args.end_epoch > scheduler.tot_epochs:
            args.end_epoch = scheduler.tot_epochs

        for epoch in range(args.start_epoch, args.end_epoch):
            dm.set_epoch(epoch)
            train(ts, epoch, dm.trn_dl, model, curr_mod, optimizer, scheduler)
            validate(ts, epoch, dm.val_dl, model)
            torch.save({'state_dict':model.module.state_dict()}, './runs/avging_checkpoint.pth.tar')
            #exit()

        # Save current 'frozen' model's weights
        model_name = './runs/model-TS'+str(ts)+'.pth.tar'
        torch.save({'state_dict':model.module.state_dict()}, model_name)

    # Save final model's weights
    torch.save({'state_dict':model.module.state_dict()}, './runs/final_avgd_model.pth.tar')

def train(timestep, epoch, train_loader, model, curr_mod, optimizer, scheduler):
    net_meter = NetworkMeter()
    timer = TimeMeter()
    obj_vals = AverageMeter()  # objective function values
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    curr_mod.eval()

    for batch_ix,(x,target) in enumerate(train_loader):
        timer.batch_start()
        scheduler.update_lr(epoch, batch_ix + 1, len(train_loader))

        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()

            x.requires_grad = True

            output = model(x)
            out_curr = curr_mod(x)

            # compute softmax-KL divergence (better than L2 distance in the large Nc case)
            output_sm = output.softmax(dim=-1)
            out_curr_sm = out_curr.softmax(dim=-1)
            obj = kl_div(output_sm.log(),out_curr_sm)     # KLDiv(f^k || x)
            #obj = kl_div(out_curr_sm.log(),output_sm)     # KLDiv(v || f^k), this may be more suitable than the line above
            obj = obj.sum(dim=-1)

            ############
            # Johnson-Lindenstrauss and finite-differences to compute grad v
            xsh = x.shape
            Nb = xsh[0]
            classes = 1000
            penalty = torch.zeros(Nb).cuda()
            num_reps = 6

            for j in range(num_reps):
                #x.requires_grad = True
                #output = model(x)

                W = torch.randn_like(output) * (1/(classes**0.5))

                wv_dot = (output * W).sum(dim=-1)

                # now do a finite-difference approximation for grad wv_dot
                grad_wv = grad(wv_dot.sum(), x, retain_graph=True)[0]


                sh = grad_wv.shape
                #x.requires_grad = False
                #x = x.detach()

                v = grad_wv.reshape(sh[0],-1)
                nv = v.norm(2,dim=-1,keepdim=True)

                v = v.div(nv)
                v = v.reshape(sh)
                dt = 0.1
                xf = x + dt*v  # forward Euler step

                outf = model(xf)

                forward = (outf * W).sum(dim=-1)
                pen = (forward - wv_dot).div(dt)
                pen = pen.pow(2)
                penalty = penalty + pen

            # put it all together into a single objective function
            objective = obj + args.gamma*0.5*(1/args.num_timesteps)*(args.std**2)*penalty
            objective = objective.mean()

            # take SGD step
            objective.backward()
            optimizer.step()

        # log results
        timer.batch_end()
        corr1, corr5 = correct(output.data, target, topk=(1, 5))
        reduced_obj, batch_total = to_python_float(objective.data), to_python_float(x.size(0))
        #if args.distributed: # Must keep track of global batch size, since not all machines are guaranteed equal batches at the end of an epoch
        #    metrics = torch.tensor([batch_total, reduced_obj, corr1, corr5]).float().cuda()
        #    batch_total, reduced_obj, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
        #    reduced_obj = reduced_obj/dist_utils.env_world_size()
        top1acc = to_python_float(corr1)*(100.0/batch_total)
        top5acc = to_python_float(corr5)*(100.0/batch_total)

        obj_vals.update(reduced_obj, batch_total)
        top1.update(top1acc, batch_total)
        top5.update(top5acc, batch_total)

        should_print = (batch_ix%args.print_freq == 0) or (batch_ix==len(train_loader))
        if should_print:
            tb.log_memory()
            tb.log_trn_times(timer.batch_time.val, timer.data_time.val, x.size(0))
            tb.log_trn_loss(obj_vals.val, top1.val, top5.val)

            recv_gbit, transmit_gbit = net_meter.update_bandwidth()
            tb.log("sizes/batch_total", batch_total)
            tb.log('net/recv_gbit', recv_gbit)
            tb.log('net/transmit_gbit', transmit_gbit)

            out = (f'Timestep: {timestep:d}\t'
                   f'Epoch: [{epoch}][{batch_ix}/{len(train_loader)}]\t'
                   f'Objective val.: {obj_vals.val:.4f} ({obj_vals.avg:.4f})\t'
                   f'Pen. val.: {args.gamma*0.5*(1/args.num_timesteps)*(args.std**2)*penalty.mean().item():.4f}\t'
                   f'Top1 acc.: {top1.val:.3f} ({top1.avg:.3f})\t'
                   f'Top5 acc.: {top5.val:.3f} ({top5.avg:.3f})')
            log.verbose(out)

        tb.update_step_count(batch_total)

        ## save checkpoint (good for warm-starting)
        #torch.save({'state_dict':model.module.state_dict()}, 'checkpoint.pth.tar')

def validate(timestep, epoch, val_loader, model):
    timer = TimeMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for batch_ix,(x,target) in enumerate(val_loader):
        timer.batch_start()
        #if args.distributed:
        #    top1acc, top5acc, batch_total = distributed_predict(x, target, model)
        #else:
        with torch.no_grad():
            output = model(x)
        batch_total = x.size(0)
        top1acc, top5acc = accuracy(output.data, target, topk=(1,5))
        timer.batch_end()

        timer.batch_end()
        top1.update(to_python_float(top1acc), to_python_float(batch_total))
        top5.update(to_python_float(top5acc), to_python_float(batch_total))
        should_print = (batch_ix%args.print_freq == 0) or (batch_ix==len(val_loader))
        if should_print:
            out = (f'VALIDATE [{epoch}][{batch_ix}/{len(val_loader)}]\t'
                   f'Top1 acc.: {top1.val:.3f} ({top1.avg:.3f})\t'
                   f'Top5 acc.: {top5.val:.3f} ({top5.avg:.3f})')
            log.verbose(out)

def distributed_predict(input, target, model):
    # Allows distributed prediction on uneven batches. Test set isn't always large enough for every GPU to get a batch
    batch_size = input.size(0)
    output = corr1 = corr5 = valid_batches = 0

    if batch_size:
        with torch.no_grad():
            output = model(input)
        # measure accuracy and record loss
        valid_batches = 1
        corr1, corr5 = correct(output.data, target, topk=(1, 5))

    metrics = torch.tensor([batch_size, valid_batches, corr1, corr5]).float().cuda()
    batch_total, valid_batches, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()

    top1 = corr1*(100.0/batch_total)
    top5 = corr5*(100.0/batch_total)
    return top1, top5, batch_total

class DataManager():
    def __init__(self, phases):
        self.phases = self.preload_phase_data(phases)
    def set_epoch(self, epoch):
        cur_phase = self.get_phase(epoch)
        if cur_phase: self.set_data(cur_phase)
        if hasattr(self.trn_smp, 'set_epoch'): self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'): self.val_smp.set_epoch(epoch)

    def get_phase(self, epoch):
        return next((p for p in self.phases if p['ep'] == epoch), None)

    def set_data(self, phase):
        """Initializes data loader."""
        if phase.get('keep_dl', False):
            log.event(f'Batch size changed: {phase["bs"]}')
            tb.log_size(phase['bs'])
            self.trn_dl.update_batch_size(phase['bs'])
            return

        log.event(f'Dataset changed.\nImage size: {phase["sz"]}\nBatch size: {phase["bs"]}\nTrain Directory: {phase["trndir"]}\nValidation Directory: {phase["valdir"]}')
        tb.log_size(phase['bs'], phase['sz'])

        self.trn_dl, self.val_dl, self.trn_smp, self.val_smp = phase['data']
        self.phases.remove(phase)

        # clear memory before we begin training
        gc.collect()

    def preload_phase_data(self, phases):
        for phase in phases:
            if not phase.get('keep_dl', False):
                self.expand_directories(phase)
                phase['data'] = self.preload_data(**phase)
        return phases

    def expand_directories(self, phase):
        trndir = phase.get('trndir', '')
        valdir = phase.get('valdir', trndir)
        phase['trndir'] = args.data+trndir+'/train'
        phase['valdir'] = args.data+valdir+'/validation'

    def preload_data(self, ep, sz, bs, trndir, valdir, **kwargs): # dummy ep var to prevent error
        if 'lr' in kwargs: del kwargs['lr'] # in case we mix schedule and data phases
        """Pre-initializes data-loaders. Use set_data to start using it."""
        if sz == 128: val_bs = max(bs, 512)
        elif sz == 224: val_bs = max(bs, 256)
        else: val_bs = max(bs, 128)
        return dataloader.get_loaders(trndir, valdir, bs=bs, val_bs=val_bs, sz=sz, workers=args.workers, distributed=False, **kwargs)

# ### Learning rate scheduler
class Scheduler():
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        if len(phase['lr']) == 2:
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase_lr(self, phase, epoch, batch_curr, batch_tot):
        lr_start, lr_end = phase['lr']
        ep_start, ep_end = phase['ep']
        if 'epoch_step' in phase: batch_curr = 0 # Optionally change learning rate through epoch step
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear_lr(lr_start, lr_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear_lr(self, lr_start, lr_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr
        step_size = (lr_end - lr_start)/step_tot
        return lr_start + step_curr * step_size

    def get_current_phase(self, epoch):
        for phase in reversed(self.phases):
            if (epoch >= phase['ep'][0]): return phase
        raise Exception('Epoch out of range')

    def get_lr(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['lr']) == 1: return phase['lr'][0] # constant learning rate
        return self.linear_phase_lr(phase, epoch, batch_curr, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if self.current_lr == lr: return
        if ((batch_num == 1) or (batch_num == batch_tot)):
            log.event(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        tb.log("sizes/lr", lr)
        tb.log("sizes/momentum", args.momentum)

def correct(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k)
    return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    corrrect_ks = correct(output, target, topk)
    batch_size = target.size(0)
    return [correct_k.float().mul_(100.0 / batch_size) for correct_k in corrrect_ks]

def to_python_float(t):
    if isinstance(t, (float, int)): return t
    if hasattr(t, 'item'): return t.item()
    else: return t[0]

def listify(p=None, q=None):
    if p is None: p=[]
    elif not isinstance(p, collections.Iterable): p=[p]
    n = q if type(q)==int else 1 if q is None else len(q)
    if len(p)==1: p = p * n
    return p

if __name__ == '__main__':
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore", category=UserWarning)
    main()
