import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .functions import *

class Linear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True,
                 n_exponent_bits=None, n_mantissa_bits=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        self.n_exponent_bits = 1
        self.n_mantissa_bits = 1

        def hookFunc(module, gradInput, gradOutput):
            newGradIn = ()
            for gi in gradInput:
                if gi is not None:
                    gi.quantize_(self.n_exponent_bits, self.n_mantissa_bits)
                newGradIn += (gi,)
            return newGradIn

        self.register_backward_hook(hookFunc) 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input.quantize_(self.n_exponent_bits, self.n_mantissa_bits)
        self.weight.quantize_(self.n_exponent_bits, self.n_mantissa_bits)
        if self.bias is not None:
            self.bias.quantize_(self.n_exponent_bits, self.n_mantissa_bits)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )