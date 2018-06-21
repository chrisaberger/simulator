import numpy as np
import math

from linear import Linear
from splittensor import SplitTensor

n_samples = 100
n_in_features = 1
n_out_features = 10
batch_size = 64
n_bits = 1

lin_layer = Linear(n_samples, batch_size, n_bits, n_in_features, n_out_features)

x = np.random.rand(n_samples, n_in_features)
y = np.random.uniform(0,1, size=(n_samples,))

num_batches = math.ceil(n_samples/batch_size)
num_epochs = 1
T = 1

for epoch in range(0, num_epochs):
    if epoch % T == 0:
        lin_layer.recenter() # var = offset + delta
        # Outer Loop
        for i in range(0, num_batches):
            inp = SplitTensor(x[i*batch_size:i*batch_size+batch_size,:])
            cur_batch_size = x[i*batch_size:i*batch_size+batch_size,:].shape[0]
            lin_layer.forward(inp, i)
            dummy_grad = np.random.rand(cur_batch_size, n_out_features)
            lin_layer.backward(dummy_grad, i)
            # lin_layer.step()
        # potentially reset scale factor here and quantize the saved value.

    # Inner Loop
    for i in range(0, num_batches):
        x_in = SplitTensor(x[i*batch_size:i*batch_size+batch_size,:])
        lin_layer.lp_forward(x_in, i)
        cur_batch_size = x[i*batch_size:i*batch_size+batch_size,:].shape[0]
        dummy_grad = np.random.rand(cur_batch_size, n_out_features)
        lin_layer.lp_backward( SplitTensor(dummy_grad), i )
        # lin_layer.step()
