import numpy as np
from splittensor import SplitTensor
from quantize import *

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

class CrossEntropy:

    def __init__(self, n_samples, out_features, batch_size):
        self.lp_fwd_outer_result = np.zeros((n_samples, out_features))
        self.batch_size = batch_size
        self.scale_factor = 1e-3
        self.num_bits = 8

    def forward(self, x, y):
        if type(x) is SplitTensor:
            x = x.data()
        self.pred = stablesoftmax(x)

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
        #q_pred = quantize(self.pred, self.num_bits, self.scale_factor)
        np.copyto(store_location, self.pred)
        return loss

    def backward_inner(self, batch_index):
        start, end = batch_index*self.batch_size, (batch_index+1)*self.batch_size
        store_location = self.lp_fwd_outer_result[start:end,]
        return SplitTensor(store_location, self.pred-store_location)

