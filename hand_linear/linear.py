from tensor import Tensor
from quantize import quantize
import numpy as np

"""
Applies a linear transformation to the incoming data: y=Wx

Parameters: 
in_features – size of each input sample
out_features – size of each output sample

Shape:
Input: (N,∗,in_features) where ∗ means any number of additional dimensions
Output: (N,∗,out_features) where all but the last dimension are the same shape 
as the input.

Variables:  
weight – the learnable weights of the module of shape 
(out_features x in_features)
"""
class Linear:
    def __init__(self, n_samples, batch_size, n_bits, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.uniform( 0,
                                                0.1, 
                                                (out_features, in_features)) )

        self.num_bits = n_bits
        self.scale_factor = 1e-2

        # Needed to 'cache' the full precision result from the outer loop. 
        self.batch_size = batch_size
        self.lp_fwd_result = np.zeros((n_samples, in_features))

    def recenter(self):
        self.weight.offset = self.weight.offset + self.weight.delta 
        self.weight.delta = np.zeros_like(self.weight.delta)
        self.weight.quantize(self.num_bits, self.scale_factor)

    def forward(self, input, index):
        """
        Computes the full precision forward value and stores it in 
        'self.fp_result' for us to use in the low precision foward.
        """
        index = index*self.batch_size
        result = np.dot(input.offset, self.weight.offset)
        self.lp_fwd_result[index:index+self.batch_size,:] = quantize( 
            np.array(result, copy=True), self.num_bits, self.scale_factor)
        return result

    def _quantize(self, np_array):
        return quantize( np_array, self.num_bits, self.scale_factor)
    
    def lp_forward(self, input, index):
        """
        We will call each term inside here 'term1', 'term2', and 'term3' in the 
        code.

         f_{k+1}(x,d1,..,dk) = g( 
            [[ ok * fk(x,\bar 0) ]] + 
            [[ ok * (fk(x,d1,...,d(k-1))) -  fk(x,\bar 0)) ]] +  
            [[dk*fk(x, \bar d)]] ) 
        """
        input.quantize(self.num_bits, self.scale_factor)

        # Should all be in LP.
        index = index*self.batch_size
        term1 = self.lp_fwd_result[index:index+self.batch_size,:]

        delta = (input.lp_offset+input.delta) - input.lp_offset
        term2 = self._quantize(np.dot(delta, self.weight.lp_offset))
        
        term3 = self._quantize( np.dot( (input.lp_offset+input.delta), 
                                        self.weight.delta ) )

        return Tensor(term1, term2+term3)
