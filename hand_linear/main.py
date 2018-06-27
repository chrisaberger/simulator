import numpy as np
import math

from linear import Linear
from splittensor import SplitTensor
from loss import *
import mnist

#mnist.init()
x_train, t_train, x_test, t_test = mnist.load()

n_samples = x_train.shape[0]
n_in_features = x_train.shape[1]
n_out_features = 10
batch_size = 100
n_bits = 1

np.random.seed(1234)
lin_layer = Linear(n_samples, batch_size, n_bits, n_in_features, n_out_features)
loss_layer = CrossEntropy()

x = x_train
y = t_train

num_batches = math.ceil(n_samples/batch_size)
num_epochs = 10
T = 1
lr = 0.001

for epoch in range(0, num_epochs):
    cost = 0
    for i in range(0, num_batches):
        batch_index = i * batch_size
        inp = SplitTensor(x[batch_index:batch_index+batch_size,:])
        cur_batch_size = x[batch_index:batch_index+batch_size,:].shape[0]
        fwd = lin_layer.forward(inp)
        cost += loss_layer.forward(fwd.offset, 
                                  y[batch_index:batch_index+batch_size,])
        lin_layer.backward(loss_layer.backward(), i)
        lin_layer.step(lr)

    test_layer = lin_layer.forward(x_test)
    predY = test_layer.argmax(axis=1)
    print("Cost: " + str(cost/num_batches) + 
          " Accuracy: " + str(100*np.mean(predY == t_test)))

exit()
for epoch in range(0, num_epochs):
    if epoch % T == 0:
        lin_layer.recenter() # var = offset + delta
        # Outer Loop
        cost = 0
        for i in range(0, num_batches):
            inp = SplitTensor(x[i*batch_size:i*batch_size+batch_size,:])
            cur_batch_size = x[i*batch_size:i*batch_size+batch_size,:].shape[0]
            fwd = lin_layer.forward(inp, i)
            cost += loss_layer.forward(fwd.offset, 
                                      y[i*batch_size:i*batch_size+batch_size,])
            lin_layer.backward(loss_layer.backward(), i)
            lin_layer.step(lr)

        #print(x_test.shape)
        test_layer = lin_layer.forward(x_test)
        predY = test_layer.argmax(axis=1)
        print("Cost: " + str(cost/num_batches) + 
              " Accuracy: " + str(100*np.mean(predY == t_test)))
        # lin_layer.step()
        # potentially reset scale factor here and quantize the saved value.
    # Inner Loop
    """
    for i in range(0, num_batches):
        x_in = SplitTensor(x[i*batch_size:i*batch_size+batch_size,:])
        lin_layer.lp_forward(x_in, i)
        cur_batch_size = x[i*batch_size:i*batch_size+batch_size,:].shape[0]
        dummy_grad = np.random.rand(cur_batch_size, n_out_features)
        lin_layer.lp_backward( SplitTensor(dummy_grad), i )
        # lin_layer.step()
    """