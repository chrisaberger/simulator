import numpy as np

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def cross_entropy(x, y):
    print(x)
    #print(y)
    pred = stablesoftmax(x)
    print(pred)
    for i in range(len(x)):
        pred[i][y[i]] = pred[i][y[i]] - 1
