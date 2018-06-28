import numpy as np
import math
from data_util import load_mnist
import torch

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def init_weights(n_in_features, n_out_features, seed=42):
    torch.manual_seed(seed)
    return torch.nn.Linear(n_in_features, n_out_features, bias=False) \
                                                        .weight.detach().numpy()
# User input parameters.
batch_size = 100
lr = 0.01
num_epochs = 100

# Load the MNIST data.
#mnist.init()
#x_train, t_train, x_test, t_test = mnist.load()
x_train, x_test, t_train, t_test = load_mnist(onehot=False)

seed = 42
n_samples = x_train.shape[0]
n_in_features = x_train.shape[1]
n_out_features = 10

np.random.seed(42)
num_batches = math.ceil(n_samples/batch_size)

avg = True
# Randomly initialize weights.
w = init_weights(n_in_features, n_out_features, seed)

for epoch in range(0, num_epochs):
    num_batches = n_samples // batch_size
    cost = 0
    gradnorm = 0
    for k in range(num_batches):
        # Get x,y data we will train from.
        start, end = k * batch_size, (k + 1) * batch_size
        x = x_train[start:end]
        y = t_train[start:end]

        # Actual work in the loop. 
        xi_dot_w = np.dot(x, w.T)
        pred = stablesoftmax(xi_dot_w)

        logsm = -np.log(pred)
        loss = 0
        for i in range(len(x)):
            loss += logsm[i, y[i]]
        loss = loss / len(x)
        cost += loss

        for i in range(len(x)):
            pred[i][y[i]] = pred[i][y[i]] - 1
        pred = pred / batch_size
        
        grad = np.dot(pred.T, x)
        # SGD Step.
        w = w - lr * grad
        #print(np.linalg.norm(grad))

    # Prediction.
    test_layer = np.dot(x_test, w.T)
    predY = test_layer.argmax(axis=1)
    print("Epoch: " + str(epoch+1) + ", cost: " + str(round(cost/num_batches, 6)) 
        + ", acc: " + str(round(100*np.mean(predY == t_test), 2)) + "%") 
