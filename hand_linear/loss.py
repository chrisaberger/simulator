import numpy as np
from splittensor import SplitTensor
from quantize import *
from interpolator import *

def stablesoftmax(x, exp_fn):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    #print(str(np.amax(shiftx)) + " " + str(np.amin(shiftx)))
    exps = exp_fn(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def lpstablesoftmax(x, iexp, num_bits, scale_factor):
    """Compute the softmax of vector x in a numerically stable way."""
    x = quantize(x, num_bits, scale_factor)
    x_max = quantize(np.max(x, axis=1), num_bits, scale_factor)
    shiftx = x - x_max.reshape((-1,1))
    #print(str(np.amax(shiftx)) + " " + str(np.amin(shiftx)))
    exps = iexp(shiftx)
    # TODO Can we cache the offset?
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
        #self.iexp.adapt_linear(-10, 10, 1.0, 1.0)
        #self.iexp.adapt_linear(-70, 100, 5.5, 5.5)
        #self.iexp.adapt_linear(0, -20, 5.5, 5.5)
        self.iexp.chunk(min = -14, max = 0, num_points = 10)

    def forward(self, x, y):
        #if type(x) is SplitTensor:
        #    x = x.data()
        self.pred = stablesoftmax(x, np.exp)

        # Compute Loss
        logsm = -np.log(self.pred)
        loss = 0
        for i in range(len(x)):
            loss += logsm[i, y[i]]
        loss = loss / len(x)

        # Compute Gradient
        for i in range(len(x)):
            self.pred[i][y[i]] = self.pred[i][y[i]] - 1
        self.pred = self.pred / len(x)
        return loss

    def forward_lp(self, x, y):
        if type(x) is SplitTensor:
            x = x.data()
            x = quantize(x, self.num_bits, self.scale_factor)
        self.pred = lpstablesoftmax(x, self.iexp.forward_fn_interpolate, 
                                    self.num_bits, self.scale_factor)
        #self.pred = stablesoftmax(x, self.iexp.forward_fn_interpolate)
        #self.pred = stablesoftmax(x, np.exp)
 
        # Compute Loss
        logsm = -np.log(self.pred)
        loss = 0
        for i in range(len(x)):
            loss += logsm[i, y[i]]
        loss = loss / len(x)

        # Compute Gradient
        for i in range(len(x)):
            self.pred[i][y[i]] = self.pred[i][y[i]] - 1
        self.pred = quantize(self.pred, self.num_bits, self.scale_factor)
        self.pred = self.pred / len(x)
        self.pred = quantize(self.pred, self.num_bits, self.scale_factor)
        return loss

    def forward_interp(self, x, y):
        if type(x) is SplitTensor:
            x = x.data()
        self.pred = stablesoftmax(x, self.iexp.forward_fn_interpolate)
        #self.pred = stablesoftmax(x, np.exp)
 
        # Compute Loss
        logsm = -np.log(self.pred)
        loss = 0
        for i in range(len(x)):
            loss += logsm[i, y[i]]
        loss = loss / len(x)

        # Compute Gradient
        for i in range(len(x)):
            self.pred[i][y[i]] = self.pred[i][y[i]] - 1
        self.pred = self.pred / len(x)
        return loss

    def backward(self):
        return self.pred

    def forward_store(self, x, y, batch_index):
        loss = self.forward(x, y)

        start, end = batch_index*self.batch_size, (batch_index+1)*self.batch_size
        store_location = self.lp_fwd_outer_result[start:end,]
        q_pred = quantize(self.pred, self.num_bits, self.scale_factor)
        np.copyto(store_location, q_pred)
        return loss

    def backward_lp(self):
        return quantize(self.pred, self.num_bits, self.scale_factor)

    def backward_inner(self, batch_index):
        start, end = batch_index*self.batch_size, (batch_index+1)*self.batch_size
        store_location = self.lp_fwd_outer_result[start:end,]
        q_diff = quantize((self.pred-store_location), 
                          self.num_bits, 
                          self.scale_factor)
        #print(str(np.amax(store_location)) + " " + str(np.amin(store_location)))
        return SplitTensor(store_location, q_diff)

