import numpy as np
import math

from data_util import load_mnist
import utils
import model
import copy

def sgd_baseline(in_data, model):
    for epoch in range(0, in_data.num_epochs):
        cost = 0
        for batch_index in range(0, in_data.num_batches):
            x, y = utils.get_data(batch_index, in_data)
            cost += model.forward(x, y)
            model.backward()
            model.step()

        predY = model.predict(in_data.x_test)
        utils.print_info(epoch, 
                         cost/in_data.num_batches, 
                         100*np.mean(predY == in_data.y_test))


def bit_centering(in_data, model):
    for epoch in range(0, in_data.num_epochs):
        if epoch % T == 0:
            model.recenter()
            cost = 0
            for batch_index in range(0, in_data.num_batches):
                x, y = utils.get_data(batch_index, in_data)
                cost += model.forward(x, y)
                model.backward()
                model.step()

            # Cache the results.
            for batch_index in range(0, num_batches):
                x, y = utils.get_data(batch_index, in_data)
                cost += model.forward_store(x, y, batch_index)
                model.backward_store(batch_index)

            predY = model.predict(in_data.x_test)
            utils.print_info(epoch, 
                             cost/in_data.num_batches, 
                             100*np.mean(predY == in_data.y_test))

        cost = 0
        for batch_index in range(0, num_batches):
            x, y = utils.get_data(batch_index, in_data)
            cost += model.forward_inner(x, y, batch_index)
            model.backward_inner(batch_index)
            model.step_inner()

        predY = model.predict_inner(in_data.x_test)
        utils.print_info(epoch, 
                         cost/in_data.num_batches, 
                         100*np.mean(predY == in_data.y_test))

batch_size = 100
num_epochs = 10
T = 10
lr = 0.01
n_classes = 10
n_bits = 8
scale_factor = 1e-1

utils.set_seed(42)
x_train, x_test, y_train, y_test = load_mnist(onehot=False)

model = model.LogisticRegression(n_samples=x_train.shape[0], 
                                 batch_size=batch_size, 
                                 n_bits=n_bits,
                                 scale_factor=scale_factor,
                                 in_features=x_train.shape[1], 
                                 out_features=n_classes,
                                 lr=lr)

num_batches = math.ceil(x_train.shape[0]/batch_size)
in_data = utils.OptimizerData(num_epochs, num_batches, batch_size, 
                              x_train, x_test, y_train, y_test)

print("SGD BASELINE")
#sgd_baseline(in_data, copy.deepcopy(model))
print()
print("BIT CENTERING")
bit_centering(in_data, copy.deepcopy(model))
