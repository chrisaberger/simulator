from splittensor import SplitTensor
from quantize import quantize
import numpy as np
import math

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
        self.lp_fwd_outer_result = np.zeros((n_samples, out_features))

        num_batches = math.ceil( n_samples / batch_size )
        self.lp_back_outer_result = np.zeros((num_batches * self.out_features, 
                                              self.in_features))

        # Needed for backwards pass
        self.saved_input = None

    def recenter(self):
        self.weight.recenter(self.num_bits, self.scale_factor)

    def _numpy_quantize(self, np_array):
        return quantize( np_array, self.num_bits, self.scale_factor)

    # TODO: This should probably go in another file as it doesn't really depend
    # on the layer and is a standalone op.
    def _lp_multiply(self, fp_result, x, y):
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

        term2 = self._numpy_quantize(np.dot(x.delta, y.lp_offset))
        
        term3 = self._numpy_quantize( np.dot( (x.lp_offset + x.delta), 
                                        y.delta ) )

        return SplitTensor(term1, term2+term3)

    def _get_data(self, data, batch_index):
        index = batch_index * self.batch_size
        return data[index:index + self.batch_size, :]

    def _save_data(self, dst_array, src_array, batch_index):
        data = self._get_data(dst_array, batch_index) 
        np.copyto( data, src_array )

    def forward(self, input, batch_index):
        """
        Computes the full precision forward value and stores it in 
        'self.fp_result' for us to use in the low precision foward.
        """

        # Needed for backwards pass
        self.saved_input = np.array(input.offset, copy = True)

        result = np.dot(input.offset, self.weight.offset.T)
        self._save_data( self.lp_fwd_outer_result,
                         self._numpy_quantize( result ),
                         batch_index )
        return SplitTensor(result)

    def backward(self, grad_output, batch_index):
        # grad out is (batch_size x out_features)
        # input is (batch_size x in_features)
        self.weight.offset_grad = np.dot( grad_output.T, self.saved_input )
        index = batch_index*self.out_features
        back_outer = self.lp_back_outer_result[index : index+self.out_features, ]
        np.copyto( back_outer,
                   self._numpy_quantize( self.weight.offset_grad ) )

    def lp_forward(self, input, batch_index):
        # Check if this is the first layer where the input needs to be 
        # quantized.
        if not input.is_quantized():
            input.quantize(self.num_bits, self.scale_factor)

        self.saved_input = input

        return self._lp_multiply( \
                        self._get_data(self.lp_fwd_outer_result, batch_index), 
                        input,
                        self.weight.T() )

    def lp_backward(self, grad_output, batch_index):
        if not grad_output.is_quantized():
            grad_output.quantize(self.num_bits, self.scale_factor)
        
        index = batch_index*self.out_features
        back_outer = self.lp_back_outer_result[index:index+self.out_features, ]
        return self._lp_multiply( \
                back_outer, 
                grad_output.T(),
                self.saved_input )
