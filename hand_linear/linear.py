from splittensor import SplitTensor
from quantize import quantize
import numpy as np
import math
import torch

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
    def __init__(self, 
                 n_samples, 
                 batch_size, 
                 n_bits, 
                 scale_factor,
                 in_features, 
                 out_features):
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = SplitTensor(self._init_weights())

        self.num_bits = n_bits
        self.fwd_scale_factor = scale_factor
        self.bck_scale_factor = 1e-3

        # Needed to 'cache' the full precision result from the outer loop. 
        self.batch_size = batch_size
        self.lp_fwd_outer_result = np.zeros((n_samples, out_features))

        num_batches = math.ceil( n_samples / batch_size )
        self.lp_back_outer_result = np.zeros((num_batches * self.out_features, 
                                              self.in_features))

        # Needed for backwards pass
        self.saved_input = None

    def _init_weights(self):
        return torch.nn.Linear(self.in_features, self.out_features, bias=False) \
                                                        .weight.detach().numpy()

    def recenter(self):
        self.weight.recenter()

    def _numpy_quantize(self, np_array, sf):
        return quantize( np_array, self.num_bits, sf)

    # TODO: This should probably go in another file as it doesn't really depend
    # on the layer and is a standalone op.
    def _lp_multiply(self, fp_result, x, y, sf):
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
        #term1 = self._numpy_quantize(np.dot(x.offset, y.offset), sf)

        term2 = self._numpy_quantize(np.dot(x.delta, y.lp_offset), sf)
        
        q_x = self._numpy_quantize(x.lp_offset + x.delta, sf)
        term3 = self._numpy_quantize( np.dot(q_x, y.delta ), sf )

        return SplitTensor(term1, term2+term3)

    def _get_data(self, data, batch_index):
        index = batch_index * self.batch_size
        return data[index:index + self.batch_size, :]

    def _save_data(self, dst_array, src_array, batch_index):
        data = self._get_data( dst_array, batch_index )
        np.copyto( data, src_array )

    def forward(self, input, train = True):
        """
        Computes the full precision forward value and stores it in 
        'self.fp_result' for us to use in the low precision foward.
        """
        if train:
            self.saved_input = input
        return np.dot(input, self.weight.offset.T)

    def backward(self, grad_output):
        # grad out is (batch_size x out_features)
        # input is (batch_size x in_features)
        self.weight.offset_grad = np.dot( grad_output.T, self.saved_input )

    def step(self, lr):
        assert(self.weight.offset_grad is not None)
        self.weight.offset = self.weight.offset - (lr * self.weight.offset_grad)

    def forward_store(self, input, batch_index):
        """
        Computes the full precision forward value and stores it in 
        'self.fp_result' for us to use in the low precision foward.
        """
        result = self.forward(input, train=False)
        q_result = self._numpy_quantize(result, self.fwd_scale_factor)
        self._save_data(self.lp_fwd_outer_result,
                        q_result,
                        batch_index)
        return result

    def backward_store(self, grad_output, batch_index):
        # grad out is (batch_size x out_features)
        # input is (batch_size x in_features)
        self.backward(grad_output)
        if batch_index is not None:
            start, end = \
                batch_index*self.out_features, (batch_index+1)*self.out_features
            back_outer = self.lp_back_outer_result[start:end, ]
            #print(str(np.amax(self.weight.offset_grad)) + " " + str(np.amin(self.weight.offset_grad)))
            q_w_offset = self._numpy_quantize( self.weight.offset_grad, self.bck_scale_factor )
            np.copyto( back_outer, q_w_offset )

    def forward_inner(self, input, batch_index):
        # Check if this is the first layer where the input needs to be 
        # quantized.
        if not input.is_quantized():
            input.quantize(self.num_bits, self.fwd_scale_factor)

        if not self.weight.is_quantized():
            self.weight.quantize(self.num_bits, self.fwd_scale_factor)

        self.saved_input = input

        return self._lp_multiply( \
                        self._get_data(self.lp_fwd_outer_result, batch_index), 
                        input,
                        self.weight.T(),
                        self.fwd_scale_factor )

    def step_inner(self, lr):
        assert(self.weight.offset_grad is not None)
        self.weight.delta = self.weight.delta - (lr * self.weight.delta_grad.data())

    def debug_backward_inner(self, grad_output, batch_index):
        self.weight.offset_grad = np.dot( grad_output.T, self.saved_input.data() )
        #print(str(np.amax(self.weight.offset_grad)) + " " + str(np.amin(self.weight.offset_grad)))

    def backward_inner(self, grad_output, batch_index):
        if not grad_output.is_quantized():
            grad_output.quantize(self.num_bits, self.bck_scale_factor)
        
        index = batch_index*self.out_features
        back_outer = self.lp_back_outer_result[index:index+self.out_features, ]
        
        self.weight.delta_grad = self._lp_multiply( \
                back_outer, 
                grad_output.T(),
                self.saved_input,
                self.bck_scale_factor )