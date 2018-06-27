import numpy as np
import math

from linear import Linear
from splittensor import SplitTensor
from loss import *
import mnist

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

# User input parameters.
batch_size = 100
lr = 0.01
num_epochs = 10

# Load the MNIST data.
mnist.init()
x_train, t_train, x_test, t_test = mnist.load()

n_samples = x_train.shape[0]
n_in_features = x_train.shape[1]
n_out_features = 10

np.random.seed(1234)
num_batches = math.ceil(n_samples/batch_size)

avg = True
# Randomly initialize weights.
w = np.random.uniform(0,
                      1.0, 
                      (n_out_features, n_in_features))
for epoch in range(0, num_epochs):
    num_batches = n_samples // batch_size
    cost = 0
    for k in range(num_batches):
        # Get x,y data we will train from.
        batch_index = k * batch_size
        x = x_train[batch_index:batch_index+batch_size,:]
        y = t_train[batch_index:batch_index+batch_size,]

        # Actual work in the loop. 
        xi_dot_w = np.dot(x, w.T)
        pred = stablesoftmax(xi_dot_w)
        for i in range(len(x)):
            pred[i][y[i]] = pred[i][y[i]] - 1
        grad = np.dot(pred.T, x) / float(len(x)) # Average Gradient.
        cost += grad.sum()
        
        # SGD Step.
        w = w - lr * grad

    # Prediction.
    test_layer = np.dot(x_test, w.T)
    predY = test_layer.argmax(axis=1)
    print("Cost: " + str(cost/num_batches) + 
          " Accuracy: " + str(100*np.mean(predY == t_test)))
