""" a bunch of different ways of classifying
    TODO: add top5 and the top1/top5 versions of Cohen randomized smoothing classification
    (this will make the certification and attack code cleaner) """

import torch
import numpy as np

def top1(output):
    return output.argmax(dim=-1).view(-1)

def margin(output,delta):
    (Nb,Nc) = output.shape
    top2, lbls = output.topk(k=2)
    cond = top2[:,0] <= top2[:,1] + delta

    pred = lbls[:,0]
    pred[cond] = Nc  # make another class for 'null'

    return pred

def dist_margin(output, r, p=2):
    (Nb,Nc) = output.shape
    Ix = torch.arange(Nb)

    bases_vectors = torch.eye(Nc)
    bases_vectors *= 3.0
    dists = torch.zeros(Nb,Nc)
    if output.is_cuda:
        bases_vectors, dists = bases_vectors.cuda(), dists.cuda()

    for i in range(Nc):
        dists[:,i] = (output - bases_vectors[i]).norm(p=p, dim=1).view(-1)

    bot1,lbls = dists.topk(k=1,largest=False)
    pred = lbls.view(-1)
    cond = bot1.view(-1) > r
    pred[cond] = Nc

    return pred
