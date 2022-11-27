#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:26:27 2022

@author: hibaterrahmendjecta
"""
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import datetime
import torch.nn as nn
import math


def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1


def sample_gaussian(shape):
    return torch.randn(*shape)


def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')


Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def exponential_CDF(t, lamb):
    return 1 - torch.exp(- lamb * t)


def get_beta(iteration, anneal, beta_min=0.0, beta_max=1.0):
    assert anneal >= 1
    beta_range = beta_max - beta_min
    return min(beta_range * iteration / anneal + beta_min, beta_max)

# noinspection PyTypeChecker
class VariancePreservingTruncatedSampling:

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., t_epsilon=1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def mean_weight(self, t):
        
        return torch.exp(-0.5 * self.integral_beta(t))

    def var(self, t):
        
        return 1. - torch.exp(- self.integral_beta(t))

    def std(self, t):
        return self.var(t) ** 0.5

    def g(self, t):
        beta_t = self.beta(t)
        return beta_t ** 0.5

    def r(self, t):
        return self.beta(t) / self.var(t)

    def t_new(self, t):
        mask_le_t_eps = (t <= self.t_epsilon).float()
        t_new = mask_le_t_eps * t_eps + (1. - mask_le_t_eps) * t
        return t_new

    def unpdf(self, t):
        t_new = self.t_new(t)
        unprob = self.r(t_new)
        return unprob

    def antiderivative(self, t):
        return torch.log(1. - torch.exp(- self.integral_beta(t))) + self.integral_beta(t)

    def phi_t_le_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.r(t_eps).item() * t

    def phi_t_gt_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.phi_t_le_t_eps(t_eps).item() + self.antiderivative(t) - self.antiderivative(t_eps).item()

    def normalizing_constant(self, T):
        return self.phi_t_gt_t_eps(T)

    def pdf(self, t, T):
        Z = self.normalizing_constant(T)
        prob = self.unpdf(t) / Z
        return prob

    def Phi(self, t, T):
        Z = self.normalizing_constant(T)
        t_new = self.t_new(t)
        mask_le_t_eps = (t <= self.t_epsilon).float()
        phi = mask_le_t_eps * self.phi_t_le_t_eps(t) + (1. - mask_le_t_eps) * self.phi_t_gt_t_eps(t_new)
        return phi / Z

    def inv_Phi(self, u, T):
        t_eps = torch.tensor(float(self.t_epsilon))
        Z = self.normalizing_constant(T)
        r_t_eps = self.r(t_eps).item()
        antdrv_t_eps = self.antiderivative(t_eps).item()
        mask_le_u_eps = (u <= self.t_epsilon * r_t_eps / Z).float()
        a = self.beta_max - self.beta_min
        b = self.beta_min
        inv_phi = mask_le_u_eps * Z / r_t_eps * u + (1. - mask_le_u_eps) * \
                  (-b + (b ** 2 + 2. * a * torch.log(
                      1. + torch.exp(Z * u + antdrv_t_eps - r_t_eps * self.t_epsilon))) ** 0.5) / a
        return inv_phi


def sample_vp_truncated_q(shape, beta_min, beta_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=0.1, beta_max=20., t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)


def get_grid(sde, input_channels, input_height, n=4, num_steps=20, transform=None, 
             mean=0, std=1, clip=True):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    if transform is not None:
        y0 = transform(y0)

    if clip:
        y0 = torch.clip(y0, 0, 1)

    y0 = y0.view(
        n, n, input_channels, input_height, input_height).permute(
        2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)

    y0 = y0.data.cpu().numpy()
    return y0
    
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.module = module
        self.decay = decay
        self.shadow_params = {}
        self.nparams = sum(p.numel() for p in module.parameters())

    def init(self):
        for name, param in self.module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def apply(self, decay=None):
        decay = self.decay if decay is None else decay
        if len(self.shadow_params) == 0:
            self.init()
        else:
            with torch.no_grad():
                for name, param in self.module.named_parameters():
                    self.shadow_params[name] -= (1 - decay) * (self.shadow_params[name] - param.data)

    def set(self, other_ema):
        self.init()
        with torch.no_grad():
            for name, param in other_ema.shadow_params.items():
                self.shadow_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module.named_parameters():
            param.data.copy_(self.shadow_params[name])

    def swap(self):
        for name, param in self.module.named_parameters():
            tmp = self.shadow_params[name].clone()
            self.shadow_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )


def create(*args):
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)


def logging(s, path='./', filename='log.txt', print_=True, log_=True):
    s = str(datetime.datetime.now()) + '\t' + str(s)
    if print_:
        print(s)
    if log_:
        assert path, 'path is not define. path: {}'.format(path)
    with open(os.path.join(path, filename), 'a+') as f_log:
        f_log.write(s + '\n')   

class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward_transform(self, x, logpx=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        if logpx is None:
            return y
        return y, logpx + self._logdetgrad(x).view(x.size(0), -1).sum(1)

    def reverse(self, y, logpy=None, **kwargs):
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logpy is None:
            return x
        return x, logpy - self._logdetgrad(x).view(x.size(0), -1).sum(1)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)
        return logdetgrad

    def __repr__(self):
        return '{name}({alpha})'.format(name=self.__class__.__name__, **self.__dict__)




