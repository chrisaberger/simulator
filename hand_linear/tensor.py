import numpy as np
from quantize import *

class Tensor:
    """
    This class is a lightweight wrapper around numpy arrays.

    Given full precision data, this tensor stores the data as offset + delta:

    data = offset + delta

    The tensor also supports quantization around various variables when needed. 
    """
    def __init__(self, data, delta = None):
        """
        Accepts as input a numpy array in the 'data' field.
        """
        self.offset = np.array(data) # full precision offset
        self.lp_offset = np.array(data) # low precision offset

        if delta is None:
            delta = np.zeros_like(self.offset)
        self.delta = delta

    def quantize(self, num_bits, scale_factor):
        self.lp_offset = quantize(self.offset, num_bits, scale_factor)
        self.delta = quantize(self.delta, num_bits, scale_factor)


