from torch.optim.optimizer import Optimizer, required
import torch
from torch.autograd import Variable
import copy, logging
import math
from .lpsgd import *

class HALP(LPSGD):
    """Implements high-accuracy low-precision algorithm.
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        T (int): number of iterations between the step to take the full grad/save w
        data_loader (DataLoader): dataloader to use to load training data
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum (default: 0)
        opt (torch.optim): optimizer to baseclass (default: SGD)
        mu (float, optional): mu hyperparameter for HALP algorithm (default: 0.1)
        bits (int, optional): number of bits to use for offset (default: 8)
        biased (bool, optional): type of rounding to use for quantization (default: unbiased)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(self.__class__, self).__init__(params, lr, momentum, dampening,
                                        weight_decay, nesterov)


        params = self.param_groups[0]['params']
        self._params = params

        self._curr_w = [p.data for p in params]
        self._curr_grad = [p.data.clone() for p in params]
        
        self._w = [p.data.clone() for p in params]
        self._w_grad = [p.data.clone() for p in params]
        self._z = [p.clone() for p in params]
        self._quant_w = [p.data.clone() for p in params]

        self._set_weights_grad(self._w, self._w_grad)

    def __setstate__(self, state):
        super(self.__class__, self).__setstate__(state)

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()

    def _set_weights_grad(self, ws, gs):
        """ Set the pointers in params to ws and gs for p.data and p.grad.data
        respectively. This allows us to avoid copying data in and out of parameters.
        """
        for idx, p in enumerate(self.param_groups[0]['params']):
            if ws is not None: p.data.copy_(ws[idx])
            if gs is not None and p.grad is not None:
                p.grad.data.copy_(gs[idx])

    def _copy_weights_grad(self, ws, gs):
        """ Set the pointers in params to ws and gs for p.data and p.grad.data
        respectively. This allows us to avoid copying data in and out of parameters.
        """
        for idx, p in enumerate(self.param_groups[0]['params']):
            if ws is not None: ws[idx].copy_(p.data)
            if gs is not None and p.grad is not None:
                gs[idx].copy_(p.grad.data)

    def _rescale(self):
        """Update scale factors for z."""
        div_factor = math.pow(2.0, self._bits-1) - 1
        for i, fg in enumerate(self._full_grad):
            self._scale_factors[i] = fg.norm() / (self._mu * div_factor)

    def _reset_z(self):
        """Set z to zero."""
        for p in self._z:
            p.fill_(0)

    def step(self):
        super(self.__class__, self).step()

    def prep_outer(self):
        self._set_weights_grad(self._w, self._w_grad)

    def recenter(self):
        #self._set_weights_grad(self._w, self._w_grad)
        """
        w <- w + fp32(z)
        """
        self._copy_weights_grad(self._w, self._w_grad)
        self._copy_weights_grad(self._curr_w, self._curr_grad)
        self._set_weights_grad(self._curr_w, self._curr_grad)
        self._reset_z()
        #self._copy_weights_grad(self._, None)

    def recenter2(self):
        for w, z in zip(self._curr_w, self._z):
            w.add_(z)
        for w, cw in zip(self._w, self._curr_w):
            w.copy_(cw)
        self._reset_z()

    def step_inner(self):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        # Set the param pointers to z to update z with step
        self._set_weights_grad(self._z, None)

        loss = super(self.__class__, self).step()

        self._copy_weights_grad(self._z, None)

        for cw, w in zip(self._curr_w, self._w):
            cw.copy_(w)
        for w, z in zip(self._curr_w, self._z):
            w.add_(z)

        self._set_weights_grad(self._curr_w, self._curr_grad)

        return loss

