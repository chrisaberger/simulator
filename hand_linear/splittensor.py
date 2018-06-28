import numpy as np
from quantize import *

class SplitTensor:
    """
    This class is a lightweight wrapper around numpy arrays. Given full 
    precision data, this tensor stores the actual data as offset + delta:

        data = offset + delta

    'SplitTensors' are the objects that are always passed on wires across layers 
    of a NN. Inside of a layer we perform operations on either the offset or 
    both the 'delta' and the 'offset' of the SplitTensor:
        (1) Outer loop. In the outer loop we compute purely on the 'offset' 
        variable. This variable here represents the full precision value of the
        underyling data.
        (2) Recentering. Recentering takes data from the 'delta' and adds it 
        back to the 'offset'. After this the 'delta' is zeroed out and the 
        'offset' represents full precision value of the underlying data (just
        like in the outer loop). We also quantize this 'offset' and store it in
        'lp_offset' (we need this for the inner loop).
        (3) Inner loop. In the inner loop we operate on 'lp_offset' and 'delta'.
        As the previous line suggests 'lp_offset' quantized version of the 
        offset. 'delta' is always the low precision delta value that we 
        accumulate during the inner loop.
    """
    def __init__(self, data, delta = None):
        """
        Accepts as input a numpy array in the 'data' field.
        """
        self.offset = np.array(data) # full precision offset
        self.lp_offset = None # low precision offset

        if delta is None:
            delta = np.zeros_like(self.offset)
        self.delta = delta

        self.offset_grad = None
        self.delta_grad = None

    def quantize(self, num_bits, scale_factor):
        self.lp_offset = quantize(self.offset, num_bits, scale_factor)
        self.delta = quantize(self.delta, num_bits, scale_factor)

    def is_quantized(self):
        return self.lp_offset is not None

    def recenter(self):
        self.offset = self.offset + self.delta 
        self.delta = np.zeros_like(self.delta)
        self.lp_offset = None

    def data(self):
        return self.offset + self.delta

    def lp_data(self):
        return self.lp_offset + self.delta

    def T(self):
        new_t = SplitTensor(self.offset.T, self.delta.T)
        if self.lp_offset is not None:
            new_t.lp_offset = np.array( self.lp_offset.T, copy = True)
        if self.offset_grad is not None:
            new_t.offset_grad = np.array( self.offset_grad.T, copy = True)
        if self.delta_grad is not None:
            new_t.delta_grad = np.array( self.delta_grad.T, copy = True)
        return new_t

    def __repr__(self):
        return f"""Offset: {self.offset}\n""" +\
               f"""LP Offset: {self.lp_offset}\n""" +\
               f"""Delta: {self.delta}\n"""

