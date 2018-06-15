import torch
from .functional import *

class Base(torch.nn.Module):
    def __init__(self):
        super(Base, self).__init__()

        self.precision_hook_fn_key = None
        self.n_exponent_bits = None
        self.n_mantissa_bits = None
        
        self.scale_factor = None
        self.n_bits = None
        self.quantize_fixed = Quantizer.quantize_fixed

    def register_fixed_precision(self, scale_factor = 1.0, num_bits=None):
        if scale_factor is not None and num_bits is not None:
            self.scale_factor = scale_factor
            self.n_bits = num_bits

        def hookFunc(module, gradInput, gradOutput):
            newGradIn = ()
            for gi in gradInput:
                if gi is not None:
                    gi.quantize_fixed_(self.scale_factor, self.n_bits)
                newGradIn += (gi,)
            return newGradIn

        keys = set(self._backward_hooks.keys())
        self.register_backward_hook(hookFunc)
        # Figure out the new key that was added and set that to
        # 'precision_hook_fn_key'.
        newkeys = set(self._backward_hooks.keys())
        diff = (newkeys-keys)
        assert(len(diff) == 1)
        old_precision_hook_fn_key = self.precision_hook_fn_key
        self.precision_hook_fn_key = diff.pop()

        # If this has been called multiple times delete the previous one.
        if old_precision_hook_fn_key is not None:
            del self._backward_hooks[old_precision_hook_fn_key]
        self.quantize_fixed = True

    def register_float_precision(self, num_bits=None, num_mantissa_bits=None):
        if num_bits is not None and num_mantissa_bits is not None:
            self.n_exponent_bits = num_bits - (num_mantissa_bits+1)
            self.n_mantissa_bits = num_mantissa_bits

        self.precision = (self.n_exponent_bits, self.n_mantissa_bits)

        def hookFunc(module, gradInput, gradOutput):
            newGradIn = ()
            for gi in gradInput:
                if gi is not None:
                    gi.quantize_float_(self.n_exponent_bits, self.n_mantissa_bits)
                newGradIn += (gi,)
            return newGradIn

        keys = set(self._backward_hooks.keys())
        self.register_backward_hook(hookFunc)
        # Figure out the new key that was added and set that to
        # 'precision_hook_fn_key'.
        newkeys = set(self._backward_hooks.keys())
        diff = (newkeys-keys)
        assert(len(diff) == 1)
        old_precision_hook_fn_key = self.precision_hook_fn_key
        self.precision_hook_fn_key = diff.pop()

        # If this has been called multiple times delete the previous one.
        if old_precision_hook_fn_key is not None:
            del self._backward_hooks[old_precision_hook_fn_key]
        self.quantize_fixed = False

    def register_precision(self):
        if self.quantize_fixed:
            self.register_fixed_precision()
        else:
            self.register_float_precision()

    def quantize(self, input):
        if self.quantize_fixed:
            return input.quantize_fixed_(self.scale_factor, self.n_bits)
        else:
            return input.quantize_float_(self.n_exponent_bits, self.n_mantissa_bits)

