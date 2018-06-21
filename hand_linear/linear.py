from splittensor import SplitTensor
from quantize import quantize
import numpy as np

class Linear:
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
    def __init__(self, n_samples, batch_size, n_bits, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = SplitTensor(np.random.uniform( 0,
                                                0.1, 
                                                (out_features, in_features)) )

        self.num_bits = n_bits
        self.scale_factor = 1e-2

        # Needed to 'cache' the full precision result from the outer loop. 
        self.batch_size = batch_size
        self.lp_fwd_outer_result = np.zeros((n_samples, in_features))
        self.lp_back_outer_result = np.zeros((n_samples, in_features))

    def recenter(self):
        self.weight.recenter(self.num_bits, self.scale_factor)

    def _numpy_quantize(self, np_array):
        return quantize( np_array, self.num_bits, self.scale_factor)

    def _lp_multiply(self, fp_result, x):
        """
        We will call each term inside here 'term1', 'term2', and 'term3' in the 
        code.

         f_{k+1}(x,d1,..,dk) = g( 
            [[ ok * fk(x,\bar 0) ]] + 
            [[ ok * (fk(x,d1,...,d(k-1))) -  fk(x,\bar 0)) ]] +  
            [[dk*fk(x, \bar d)]] ) 
        """
        # Should all be in LP.
        term1 = fp_result

        term2 = self._numpy_quantize(np.dot(x.delta, self.weight.lp_offset))
        
        term3 = self._numpy_quantize( np.dot( (x.lp_offset + x.delta), 
                                        self.weight.delta ) )

        return SplitTensor(term1, term2+term3)
    def forward(self, input, batch_index):
        """
        Computes the full precision forward value and stores it in 
        'self.fp_result' for us to use in the low precision foward.
        """
        result = np.dot(input.offset, self.weight.offset)
        index = batch_index * self.batch_size
        self.lp_fwd_outer_result[index:index+self.batch_size,:] = \
                             self._numpy_quantize( np.array(result, copy=True) )
        return result

    def backward(self, grad_in):
        #grad_in * transpose(self.weight) 
        print("HERE")


    def lp_forward(self, input, index):
        """
        We will call each term inside here 'term1', 'term2', and 'term3' in the 
        code.

         f_{k+1}(x,d1,..,dk) = g( 
            [[ ok * fk(x,\bar 0) ]] + 
            [[ ok * (fk(x,d1,...,d(k-1))) -  fk(x,\bar 0)) ]] +  
            [[dk*fk(x, \bar d)]] ) 
        """

        # Check if this is the first layer where the input needs to be 
        # quantized.
        if not input.is_quantized():
            input.quantize(self.num_bits, self.scale_factor)

        return self._lp_multiply( \
                        self.lp_fwd_outer_result[index:index+self.batch_size,:], 
                        input )
