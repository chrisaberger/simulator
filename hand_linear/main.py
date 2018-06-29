import numpy as np
import math

from data_util import load_mnist
import utils
import model
import copy

from quantize import *
from interpolator import *

"""
iexp = Interpolator(np.exp)
iexp.adapt_linear(-100, 70, 5.5, 5.5)
iexp.plot()
a = np.array([-90.0, 5.0, 1.0, 0.5, 7.8, 60.9, 80.9])
print(iexp.forward(a))
print(np.exp(a))
exit()
"""

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

def lp_sgd_baseline(in_data, model):
    for epoch in range(0, in_data.num_epochs):
        cost = 0
        for batch_index in range(0, in_data.num_batches):
            x, y = utils.get_data(batch_index, in_data)
            cost += model.forward_lp(x, y)
            model.backward_lp()
            model.step()

        predY = model.predict(in_data.x_test)
        utils.print_info(epoch, 
                         cost/in_data.num_batches, 
                         100*np.mean(predY == in_data.y_test))

def sgd_bitcentering(in_data, model):
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


def svrg_baseline(in_data, model):
    w_tilde = None
    g_tilde = None
    model.lr *= 2
    for epoch in range(0, in_data.num_epochs):
        if epoch % T == 0:
            w_tilde = np.copy(model.lin_layer.weight.data())
            cost  = model.forward(in_data.x_train, in_data.y_train)
            model.backward()
            g_tilde = np.copy(model.lin_layer.weight.offset_grad)

        cost = 0
        for batch_index in range(0, in_data.num_batches):
            x, y = utils.get_data(batch_index, in_data)

            w_offset = np.copy(model.lin_layer.weight.offset)
            np.copyto(model.lin_layer.weight.offset, w_tilde)

            model.forward(x, y)
            model.backward()
            w_tilde_grad = np.copy(model.lin_layer.weight.offset_grad)

            np.copyto(model.lin_layer.weight.offset, w_offset)
            cost += model.forward(x, y)
            model.backward()
            model.step_svrg(w_tilde_grad, g_tilde)

        predY = model.predict(in_data.x_test)
        utils.print_info(epoch, 
                         cost/in_data.num_batches, 
                         100*np.mean(predY == in_data.y_test))


def svrg_bitcentering(in_data, model):
    w_tilde = None
    g_tilde = None
    model.lr *= 2

    cost = 0
    for batch_index in range(0, in_data.num_batches):
        x, y = utils.get_data(batch_index, in_data)
        cost += model.forward(x, y)
        model.backward()
        model.step()

    for epoch in range(0, in_data.num_epochs):
        if epoch % T == 0:
            model.recenter()
            cost  = model.forward(in_data.x_train, in_data.y_train)
            model.backward()
            g_tilde = np.copy(model.lin_layer.weight.offset_grad)
            w_tilde = np.copy(model.lin_layer.weight.data())

            # Cache the results.
            for batch_index in range(0, num_batches):
                x, y = utils.get_data(batch_index, in_data)
                model.forward_store(x, y, batch_index)
                model.backward_store(batch_index)

        cost = 0
        for batch_index in range(0, in_data.num_batches):
            x, y = utils.get_data(batch_index, in_data)

            index = batch_index*n_classes
            w_tilde_grad = model.lin_layer.lp_back_outer_result[index:index+n_classes, ]

            #cost += model.forward(x, y)
            #model.backward()

            cost += model.forward_inner(x, y, batch_index)
            model.backward_inner(batch_index)

            model.step_svrg(w_tilde_grad, g_tilde)

        predY = model.predict(in_data.x_test)
        utils.print_info(epoch, 
                         cost/in_data.num_batches, 
                         100*np.mean(predY == in_data.y_test))

batch_size = 100
num_epochs = 10
T = 1
lr = 0.1
n_classes = 10
n_bits = 4
fwd_scale_factor = 10e-1
bck_scale_factor = 10e-3

utils.set_seed(42)
x_train, x_test, y_train, y_test = load_mnist(onehot=False)

model = model.LogisticRegression(n_samples=x_train.shape[0], 
                                 batch_size=batch_size, 
                                 n_bits=n_bits,
                                 fwd_scale_factor=fwd_scale_factor,
                                 bck_scale_factor=bck_scale_factor,
                                 in_features=x_train.shape[1], 
                                 out_features=n_classes,
                                 lr=lr)

num_batches = math.ceil(x_train.shape[0]/batch_size)
in_data = utils.OptimizerData(num_epochs, num_batches, batch_size, 
                              x_train, x_test, y_train, y_test)

#print("SVRG BASELINE")
#svrg_baseline(in_data, copy.deepcopy(model))
#print()

print("SVRG BITCENTERING")
svrg_bitcentering(in_data, copy.deepcopy(model))

#print("SGD BASELINE")
#sgd_baseline(in_data, copy.deepcopy(model))
#print()
#print("LP SGD BASELINE")
#lp_sgd_baseline(in_data, copy.deepcopy(model))
#print()
#print("BIT CENTERING")
#sgd_bitcentering(in_data, copy.deepcopy(model))
