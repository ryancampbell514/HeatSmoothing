import argparse
import yaml
import ast
import os, sys

import torch
import torch.nn as nn

from HeatSmoothing.cifar10.train_utils import dataloader
from HeatSmoothing.cifar10.train_utils import cvmodels as models

def get_loader(model_dir, mode='test', batch_size=100, workers=4, has_cuda=True):
    state_dict = yaml.load(open(os.path.join(model_dir, 'args.yaml'), 'r'))
    state = argparse.Namespace()
    state.__dict__ = state_dict



    loader = getattr(dataloader, state.dataset)(state.datadir,
            mode=mode, transform=False,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            pin_memory=has_cuda)
    return loader



def get_model(model_dir, classes, pth_name='best.pth.tar',
        strict=True, has_cuda=True, **kwargs):
    state_dict = yaml.load(open(os.path.join(model_dir, 'args.yaml'), 'r'))
    state = argparse.Namespace()
    state.__dict__ = state_dict


    # Load model
    model_args = ast.literal_eval(state.model_args)
    model_args.update(bn=state.bn, classes=classes, bias=state.bias,
                      kernel_size=state.kernel_size, 
                      softmax=False,
                      dropout=state.dropout)
    model_args.update(**kwargs)
    if state.dataset in ['cifar10','cifar100','Fashion']:
        m = getattr(models.cifar, state.model)(**model_args)
    elif state.dataset=='TinyImageNet':
        m = getattr(models.tinyimagenet, state.model)(**model_args)
    elif state.dataset=='mnist':
        m = getattr(models.mnist, state.model)(**model_args)
    m_pth = os.path.join(model_dir, pth_name)
    state_dict = torch.load(m_pth, map_location='cpu')['state_dict']

    m.load_state_dict(state_dict, strict=strict)
    m.eval()

    for p in m.parameters():
        p.requires_grad_(False)

    if has_cuda:
        m = m.cuda()

    return m
