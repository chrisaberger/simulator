import numpy as np

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def cross_entropy(x, y):
    pred = stablesoftmax(x)
    for i in range(len(x)):
        pred[i][y[i]] = pred[i][y[i]] - 1
    return pred

class CrossEntropy:
    def forward(self, x, y):
        self.pred = stablesoftmax(x)
        for i in range(len(x)):
            self.pred[i][y[i]] = self.pred[i][y[i]] - 1
        return self.pred.sum()/len(x)

    def backward(self):
        return self.pred
