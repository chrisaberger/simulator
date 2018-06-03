import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from traverse import *
from data_util import load_mnist


def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=True))
    #model.add_module("linear2",
    #                 torch.nn.Linear(output_dim, 1, bias=False))
    return model


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    summary = traverse(fx, params=dict(model.named_parameters()), 
                      bytes_per_elem=4, log_file="fx.log")
    
    print("\n" + str(summary))

    summary = traverse(output, params=dict(model.named_parameters()), 
                      bytes_per_elem=4, log_file="output_fwd.log")
    
    print("\n" + str(summary))

    # Backward
    output.backward()

    summary = traverse(output, params=dict(model.named_parameters()), 
                      bytes_per_elem=4, log_file="output_bckwrd.log")

    print("\n" + str(summary))

    exit()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


def main():
    torch.manual_seed(42)
    trX, teX, trY, teY = load_mnist(onehot=False)
    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()

    n_examples, n_features = trX.size()
    n_classes = 10
    print("n_features: " + str(n_features) + " n_classes: " + str(n_classes))
    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    batch_size = 100

    for i in range(100):
        cost = 0.
        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer,
                          trX[start:end], trY[start:end])
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%"
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))


if __name__ == "__main__":
    main()