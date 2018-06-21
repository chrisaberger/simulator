import numpy as np

def quantize(data, num_bits, scale_factor, biased=False):
    if not biased:
        random_data = np.random.uniform(0, 1, size=data.shape)
        data = np.floor((data/float(scale_factor)) + random_data)
    else:
        data = np.floor(data/float(scale_factor) + 0.5)
    min_value = -1 * (2**(num_bits-1))
    max_value = 2**(num_bits-1) - 1
    data = np.clip(data, min_value, max_value)
    return data*scale_factor