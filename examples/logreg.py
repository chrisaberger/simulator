import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from data_util import load_mnist
import hazysim as sim

def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    
    lin_layer = sim.nn.Linear(input_dim, output_dim, bias=False)

    model.add_module("linear", lin_layer)
    return model

def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    #print(x)
    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data.item()

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

def main():
    torch.manual_seed(42)
    # Sets the global precision level (if you set per layer this is overridden
    # for said layer).
    #sim.nn.Quantizer.set_float_precision(num_bits=32, num_mantissa_bits=11)
    sim.nn.Quantizer.set_fixed_precision(scale_factor=1e-2, num_bits=8)
    trX, teX, trY, teY = load_mnist(onehot=False)
    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()

    n_examples, n_features = trX.size()
    n_classes = 10
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