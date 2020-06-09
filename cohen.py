""" code used by Cohen (2019) for certification """

import torch
import numpy as np
import time
from scipy import stats
from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint

def certify(model, x, std, is_cohen, classes, n0=100, n=100000, alpha=0.001, rule='top1'):
    """ certify (get radius) ONE IMAGE AT A TIME """

    has_cuda = x.is_cuda

    model.eval()
    xsh = x.shape  # channels x height x width

    t1 = time.time()  # record time it takes to classify
    if is_cohen:
        # draw samples of f(x+ epsilon)
        noise = torch.randn(n0,*xsh) * std
        if has_cuda:
            noise = noise.cuda()
        xn = x.unsqueeze(0) + noise  # n0 x channels x height x width
        f = model(xn)
        f_pred = f.argmax(dim=-1)  # a bunch of labels
        counts = torch.zeros(classes) #.cuda()
        if has_cuda:
            counts = counts.cuda()
        for idx in f_pred:
            counts[idx] += 1

        # get the estimated class
        if rule=='top1':
            cAHat = counts.argmax().item()
        elif rule=='top5':
            cAHat = counts.topk(k=5)[1]

    else:
        output = model(x.unsqueeze(0))
        if rule=='top1':
            cAHat = output.view(-1).argmax().item()
        elif rule=='top5':
            cAHat = output.view(-1).topk(k=5)[1]
    t2 = time.time()
    t = t2-t1

    # now get radius
    noise = torch.randn(n,*xsh) * std
    if has_cuda:
        noise = noise.cuda()
    xn = x.unsqueeze(0) + noise  # n x channels x height x width
    f = model(xn)
    f_pred = f.argmax(dim=-1)  # a bunch of labels
    if rule=='top1':
        nA = (f_pred == cAHat).float().sum().int().item()   # numer of times prediction matches initial prediction
    elif rule=='top5':
        nA = 0
        for i in range(n):
            if f_pred[i] in cAHat:
                nA += 1

    pABar = proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0]

    # return predicted class, certified accurary, and radius
    if pABar < 0.5:
        if rule=='top1':
            return -1, 0.0, t  # ABSTAIN
        elif rule=='top5':
            return torch.tensor([-1,-1,-1,-1,-1]).cuda(), 0.0, t
    else:
        radius = std * stats.norm.ppf(pABar)
        return cAHat, radius, t  # estimates class and radius

def predict(model, x, std, num_samples=1000, criterion='top1'):
    """ take in a model and images, output predicted labels
        THIS IS BATCH-WISE, WHICH IS NOT DONE IN COHEN

        mnist:   std = 0.3,
        cifar10: std = 0.1

        num_samples = 1,000
    """

    xsh = x.shape
    x.unsqueeze_(1)
    noise = torch.randn(xsh[0],num_samples,*xsh[1:])*std
    noise = noise.cuda(x.device)

    xn = x + noise
    xn = xn.view(xsh[0]*num_samples,*xsh[1:])
    f = model(xn)

    if criterion=='top1':
        preds = f.argmax(dim=-1)
    else:
        raise ValueError('please select a valid classification criterion')

    preds = preds.view(xsh[0],num_samples).mode(dim=-1)

    return preds
