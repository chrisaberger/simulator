import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class FLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, interp):
        ctx.save_for_backward(input)

        # This only works for positive numbers (as does log).
        sign, exponent, mantissa = input.break_down_fp_()
        log2_mantissa = interp(mantissa)
        
        # For Debug purposes if you want to remove the interpolator.
        # log2_mantissa = torch.log2(mantissa)
        log2 = log2_mantissa + exponent.double()
        log2_e = 1.4426950408889634
        return log2/log2_e

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 1/input
        grad_input.quantize_()
        return (grad_input.float()*grad_output.float(), None)
