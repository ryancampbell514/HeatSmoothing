import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math

# Parse command line arguments
#from flashlight.experiment_template.argparser import parser
#args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# DEFINE ALL THE CRAZY LOSS FUNCTIONS
# KL Loss
def KL_loss(output, target):
    output = F.softmax(output, dim=1)

    (Nb,Nc) = output.shape
    Ix = torch.arange(Nb)
    if has_cuda:
        Ix = Ix.cuda()
    loss = -torch.log(output[Ix,target])
    #loss = (-(output[Ix,target] - output.exp().sum(dim=1).log())).mean()
    return loss

# An unbiased loss that uses softmax probabilities
def prob_loss(output, target, epoch):
    (Nb,Nc) = output.shape
    Ix = torch.arange(Nb)
    if has_cuda:
        Ix = Ix.cuda()

    output = F.softmax(output, dim=1)

    #calculate p_k
    p_k = output[Ix,target]
    p_k = p_k.view(-1)
    if has_cuda:
        p_k = p_k.cuda()

    # calculate p_max
    top2 = torch.topk(output, 2, dim=1)[0]
    p_max = top2[:,0]
    cond = p_max == p_k
    p_max[cond] = (top2[:,1])[cond]
    #p_max = (output.max(dim=1)[0]).view(-1).float()
    if has_cuda:
        p_max = p_max.cuda()
    w = args.loss_weight
    #loss = (-torch.log((w/2)*(1+p_k-p_max) + (1-w)*p_k)).mean()
    #loss = (-torch.log(w*p_k + (1-w)*(1-p_max))).mean()
    if epoch <= 75:
        loss = (1-p_k)
    elif 75 < epoch <= 125:
        loss = (1 + torch.exp(10*(p_k-0.5))).pow(-1)
    elif epoch > 125:
        loss = (1 + torch.exp(100*(p_k-0.5))).pow(-1)
    if has_cuda:
        loss = loss.cuda()
    return loss

# Chris' new loss
def chris_loss(z, y):
    (Nb, Nc) = z.shape
    Ix = torch.arange(Nb)

    # one-hot encode the targets
    Y = torch.zeros(Nb,Nc)
    Y[Ix, y] = 1

    # one-hot encode the output
    a = z.argmax(dim=1).view(-1)
    PZ = torch.zeros(Nb,Nc)
    PZ[Ix,a] = 1

    if has_cuda:
        Y, PZ = Y.cuda(), PZ.cuda()

    #tau = args.loss_weight
    #if 15 <= epoch < 30:
    #    tau = args.loss_weight*10
    #elif 30 <= epoch < 40:
    #    tau = args.loss_weight*100
    #elif epoch >= 40:
    #    tau = args.loss_weight*1000

    #tau = 0.5
    #lam = torch.max(1 - tau/(z - PZ).norm(p=2, dim=1), torch.zeros(Nb).cuda())
    #print(lam.mean())
    #lam = lam.unsqueeze_(1)

    #lam = ((0.01 - 1)/(args.epochs - 1))*epoch + 1
    #print(lam)
    lam = args.loss_weight
    Prox = lam*z + (1-lam)*PZ

    loss = (0.5*torch.pow((Y - Prox).norm(p=2, dim=1), 2))
    #print(loss)
    #exit()
    return loss

# Distance loss
def dist_loss(output, target):
    (Nb,Nc) = output.shape
    Ix = torch.arange(Nb)

    bases_vectors = torch.eye(Nc)
    bases_vectors *= 3.0
    if has_cuda:
        bases_vectors = bases_vectors.cuda()
    dists = torch.zeros(Nb,Nc)
    for i in range(Nc):
        dists[:,i] = (output - bases_vectors[i]).norm(p=2, dim=1).view(-1)
    if has_cuda:
        Ix,dists = Ix.cuda(), dists.cuda()

    d_k = dists[Ix,target]
    d_k = d_k.view(-1)
    if has_cuda:
        d_k = d_k.cuda()

    loss = d_k.pow(2)
    #loss = d_k.pow(2)/(torch.pow(torch.max(2**0.5 - d_k, torch.tensor([0.001]*len(d_k)).cuda()), 0.5))
    #smooth_max = ((2**0.5 - d_k)*((10*(2**0.5 - d_k)).exp()) + 0.001*math.exp(10*0.001))/((10*(2**0.5 - d_k)).exp() + math.exp(10*0.001))
    #loss = d_k.pow(2)/(smooth_max**0.5)
    return loss

def cw_loss(output, target):
    (Nb,Nc) = output.shape
    Ix = torch.arange(Nb)

    #probs = F.softmax(output, dim=1)
    probs = output

    p_k = probs[Ix, target]

    # calculate p_max
    top2 = torch.topk(probs, 2, dim=1)[0]
    p_max = top2[:,0]
    cond = p_max == p_k
    p_max[cond] = (top2[:,1])[cond]
    p_max = p_max.view(-1)

    loss = (p_max - p_k)
    if has_cuda:
        loss = loss.cuda()
    return loss

def gauss_loss(output, target, means, inv_covs):
    (Nb,Nc) = output.shape
    loss = torch.zeros(Nb)
    for i in range(Nb):
        y = target[i].item()
        #print(type(output[i]))
        #print(type(means[y]))
        #print(type(inv_covs[y]))
        #exit()
        loss[i] = torch.dot(torch.mv(torch.t(inv_covs[y]), output[i] - means[y]), output[i] - means[y])
    if has_cuda:
        loss = loss.cuda()
    return loss

def boundary_loss(f, x, target):
    """ f -> output, x -> batch of images, target -> true classifications  """
    (Nb,Nc) = f.shape
    Ix = torch.arange(Nb)

    # Term 1
    bases_vectors = torch.eye(Nc)
    bases_vectors *= 3.0
    if has_cuda:
        bases_vectors = bases_vectors.cuda()
    dists = torch.zeros(Nb,Nc)
    for i in range(Nc):
        dists[:,i] = (f - bases_vectors[i]).norm(p=2, dim=1).view(-1)
    if has_cuda:
        Ix,dists = Ix.cuda(), dists.cuda()

    d_k = dists[Ix,target]
    d_k = d_k.view(-1)
    if has_cuda:
        d_k = d_k.cuda()

    # Term 2
    imshape = x[0].shape
    one_hot = torch.ones(Nb,Nc)
    one_hot[Ix,target] = 0.0
    cond = one_hot == 1

    f_k = f[Ix,target]
    f_i = f[cond].reshape(Nb,Nc-1)
    #numerator
    num = f_k.unsqueeze(1) - f_i

    f = f.mean(dim=0)
    f_grad = torch.zeros(Nc,Nb,*imshape)
    for i in range(Nc):
        f_grad[i] = torch.autograd.grad(f[i], x, retain_graph=True)[0]

    f_grad = f_grad.reshape(Nc,Nb,-1)
    f_grad = f_grad.transpose(0,1)

    f_grad_k = f_grad[Ix,target]
    f_grad_i = f_grad[cond].reshape(Nb,Nc-1,-1)

    diff = f_grad_i - f_grad_k.unsqueeze(1)
    #denominator
    denom = diff.norm(p=2,dim=-1)

    if has_cuda:
        num, denom = num.cuda(), denom.cuda()
    term2 = (num/(10e-6 + denom)).sum(dim=-1)

    #print(d_k.pow(2).mean())
    #print(10e-7*term2.mean())
    #exit()

    # put together
    #lx = (d_k.pow(2)) - ((10e-6)/2)*term2
    lx = (d_k.pow(2)) - (10e-7)*term2

    zero = torch.zeros(Nb)
    if has_cuda:
        zero = zero.cuda()
    lx = torch.max(zero, lx)
    #print(lx)
    #exit()
    return lx

def margin_loss(output, target):
    (Nb,Nc) = output.shape
    Ix = torch.arange(Nb)

    ## MARGIN LOSS FROM NOTE
    #c = 1.0
    #a = 3/(3+c)
    #epsilon = 1.0

    #print(target[0])
    #print(output[0])
    #T = output
    #T[Ix,target] -= c
    #print(T[0])
    #exit()

    #g_eps = T.div(epsilon).exp().sum(dim=-1).log().mul(epsilon)
    #lx = a*g_eps

    ## MARGIN LOSS FROM EMAIL
    rho = args.loss_weight
    #rho = 40
    f_k = output[Ix,target]

    top2 = torch.topk(output, 2, dim=1)[0]
    f_max = top2[:,0]
    cond = f_max == f_k
    f_max[cond] = (top2[:,1])[cond]
    f_max = f_max.view(-1)

    val = 1- ((f_k - f_max)/rho)
    lx = torch.max(val,torch.zeros(len(val)).cuda())
    if has_cuda:
        lx = lx.cuda()

    return lx
