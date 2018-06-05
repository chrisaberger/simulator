import torch

class BaseLayer(torch.nn.Module):

    def register_precision(self, num_bits=None, num_mantissa_bits=None):
        self.n_exponent_bits = None
        self.n_mantissa_bits = None
        if num_bits is not None and num_mantissa_bits is not None:
            self.n_exponent_bits = num_bits - (num_mantissa_bits+1)
            self.n_mantissa_bits = num_mantissa_bits

        self.precision = (self.n_exponent_bits, self.n_mantissa_bits)

        def hookFunc(module, gradInput, gradOutput):
            newGradIn = ()
            for gi in gradInput:
                if gi is not None:
                    gi.quantize_(self.n_exponent_bits, self.n_mantissa_bits)
                newGradIn += (gi,)
            return newGradIn

        self.register_backward_hook(hookFunc) 