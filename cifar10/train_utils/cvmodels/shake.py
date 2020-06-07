import torch as th
from torch.autograd import Function
from torch.distributions import Dirichlet


class Shake(Function):

    @staticmethod
    def forward(ctx, x, dim, batchdim, concentration, train):
        ctx.dim = dim
        ctx.batchdim = batchdim
        ctx.concentration = concentration
        xsh, ctx.xsh = [x.shape]*2
        ctx.dev = x.device
        ctx.train = True

        if train:
            # Randomly sample from Dirichlet distribution
            dist = Dirichlet(th.full((xsh[dim],), concentration))
            alpha = dist.sample(sample_shape=th.Size([xsh[batchdim]]))
            alpha = alpha.to(th.device(x.device))
            sh = [1 for _ in range(len(xsh))]
            sh[batchdim], sh[dim] = xsh[batchdim], xsh[dim]
            alpha = alpha.view(*sh)


            y = (x * alpha).sum(dim)
        else:
            y = x.mean(dim)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input=None

        if not ctx.train:
            raise RuntimeError('Running backward on shake when train is False')

        if ctx.needs_input_grad[0]:
            dist = Dirichlet(th.full((ctx.xsh[ctx.dim],),ctx.concentration))
            beta = dist.sample(sample_shape=th.Size([ctx.xsh[ctx.batchdim]]))
            beta = beta.to(th.device(ctx.dev))
            sh = [1 for _ in range(len(ctx.xsh))]
            sh[ctx.batchdim], sh[ctx.dim] = ctx.xsh[ctx.batchdim], ctx.xsh[ctx.dim]
            beta = beta.view(*sh)
            grad_output = grad_output.unsqueeze(ctx.dim).expand(*ctx.xsh)

            grad_input = grad_output * beta

        return grad_input, None, None, None, None, None

shake = Shake.apply
