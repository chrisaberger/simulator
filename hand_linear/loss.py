import numpy as np
from splittensor import SplitTensor

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

class CrossEntropy:
    def forward(self, x, y):
        if type(x) is SplitTensor:
            x = x.offset + x.delta
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
