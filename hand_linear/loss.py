import numpy as np
from splittensor import SplitTensor
from quantize import *
from interpolator import *

def stablesoftmax(x, exp_fn):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    #DEBUG print(str(np.amax(shiftx)) + " " + str(np.amin(shiftx)))
    exps = exp_fn(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def lpstablesoftmax(x, iexp, num_bits, scale_factor):
    """Compute the softmax of vector x in a numerically stable way."""
    x = quantize(x, num_bits, scale_factor)
    x_max = quantize(np.max(x, axis=1), num_bits, scale_factor)
    shiftx = x - x_max.reshape((-1,1))
    #DEBUG print(str(np.amax(shiftx)) + " " + str(np.amin(shiftx)))
    exps = iexp(shiftx)
    exps = quantize(exps, num_bits, scale_factor)
    sum_exps = quantize(np.sum(exps, axis=1).reshape(-1,1), 
                        num_bits, scale_factor)
    return quantize(exps / sum_exps, num_bits, scale_factor)

class CrossEntropy:

    def __init__(self, n_samples, out_features, batch_size, num_bits):
        self.lp_fwd_outer_result = np.zeros((n_samples, out_features))
        self.batch_size = batch_size
        self.scale_factor = 1e-3
        self.num_bits = num_bits

        self.iexp = Interpolator(np.exp)
        self.iexp.chunk(min=-14, max=0, num_points=10)

    def _get_loss(self, x, y):
        logsm = -np.log(self.pred)
        loss = 0
        for i in range(len(x)):
            loss += logsm[i, y[i]]
        return loss / len(x)

    def _compute_grad(self, x, y, low_precision=False):
        """
        Computes the gradient and stores it in 'self.pred'.
        """
        for i in range(len(x)):
            self.pred[i][y[i]] = self.pred[i][y[i]] - 1
        if low_precision:
            self.pred = quantize(self.pred, self.num_bits, self.scale_factor)
        self.pred = self.pred / len(x)
        if low_precision:
            self.pred = quantize(self.pred, self.num_bits, self.scale_factor)

    def forward(self, x, y):
        self.pred = stablesoftmax(x, np.exp)

        # Compute Loss
        loss = self._get_loss(x, y)

        # Compute Gradient
        self._compute_grad(x, y)

        return loss

    def forward_lp(self, x, y):
        if type(x) is SplitTensor:
            x = x.data()
            x = quantize(x, self.num_bits, self.scale_factor)
        self.pred = lpstablesoftmax(x, self.iexp.forward_fn_interpolate, 
                                    self.num_bits, self.scale_factor)

        # Compute Loss
        loss = self._get_loss(x, y)

        # Compute Gradient
        self._compute_grad(x, y, low_precision=True)

        return loss

    def forward_interp(self, x, y):
        if type(x) is SplitTensor:
            x = x.data()
        self.pred = stablesoftmax(x, self.iexp.forward_fn_interpolate)
 
        # Compute Loss
        loss = self._get_loss(x, y)

        # Compute Gradient
        self._compute_grad(x, y)

        return loss

    def forward_store(self, x, y, batch_index):
        loss = self.forward(x, y)

        start, end = batch_index*self.batch_size, (batch_index+1)*self.batch_size
        store_location = self.lp_fwd_outer_result[start:end,]
        q_pred = quantize(self.pred, self.num_bits, self.scale_factor)
        np.copyto(store_location, q_pred)
        return loss

    def backward(self):
        return self.pred

    def backward_lp(self):
        return quantize(self.pred, self.num_bits, self.scale_factor)

    def backward_inner(self, batch_index):
        start, end = batch_index*self.batch_size, \
                     (batch_index+1)*self.batch_size
        store_location = self.lp_fwd_outer_result[start:end,]
        q_diff = quantize( ( self.pred-store_location ) , 
                             self.num_bits, 
                             self.scale_factor)
        #print(str(np.amax(store_location)) + " " + str(np.amin(store_location)))
        return SplitTensor(store_location, q_diff)

