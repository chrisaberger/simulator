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

        #for w in self._w:
        #    w[w == w] = 0.69
        #self._w_grad[self._w_grad == self._w_grad] = 0.69
        #self._set_weights_grad(self._w, self._w_grad)
        self._set_weights_grad(self._w, self._w_grad, self._w, self._w_grad)


    def __setstate__(self, state):
        super(self.__class__, self).__setstate__(state)

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()

    def _set_weights_grad(self, wssrc, gssrc, wsdst=None, gsdst=None):
        """ Set the pointers in params to ws and gs for p.data and p.grad.data
        respectively. This allows us to avoid copying data in and out of parameters.
        """
        for idx, p in enumerate(self._params):
            if wsdst is not None:
                wsdst[idx].data.copy_(p.data)
            if wssrc is not None: 
                p.data.copy_(wssrc[idx].data)
            if gsdst is not None and p.grad is not None:
                gsdst[idx].data.copy_(p.grad.data)
            if gssrc is not None and p.grad is not None:
                p.grad.data.copy_(gssrc[idx].data)
                #assert (p.grad.data.data_ptr() == gs[idx].data_ptr())

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
        self._set_weights_grad(self._w, None, self._w)
        super(self.__class__, self).step()

    def recenter(self):
        #self._set_weights_grad(self._w, self._w_grad)
        """
        w <- w + fp32(z)
        """
        for w, z in zip(self._w, self._z):
            z.quantize_()
            w.add_(z)
        """
        z <- 0
        """
        self._reset_z()
        """
        w_hat <- quant(w)
        """
        for p, p0 in zip(self._quant_w, self._w):
            p.copy_(p0)
            p.quantize_()

    def step_inner(self):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        # Set the param pointers to z to update z with step
        self._set_weights_grad(self._z)

        loss = super(self.__class__, self).step()

        # curr_w = quant(w) + z
        for cw, qw in zip(self._curr_w, self._quant_w):
            cw.copy_(qw)
        for cw, z in zip(self._curr_w, self._z):
            z.quantize_()
            cw.add_(z)
            cw.quantize_()

        # Update param pointers to curr_w for user access
        self._set_weights_grad(self._curr_w)

        return loss