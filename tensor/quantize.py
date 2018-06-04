
import math
import torch
import struct
import ctypes
import sys
import logging
import numpy as np

# Modified from
# https://github.com/aaron-xichen/pytorch-playground/blob/master/utee/quant.py

def quantize_(input, scale_factor, bits, biased=False):
    assert bits >= 1, bits
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    if biased:
      adj_val = 0.5
    else:
      # Generate tensor of random values from [0,1]
      adj_val = torch.Tensor(input.size()).type(input.type()).uniform_()

    rounded = input.div_(scale_factor).add_(adj_val).floor_()
    clipped_value = rounded.clamp_(min_val, max_val)
    clipped_value *= scale_factor

def saturate_(input, scale_factor, bits):
    bound = math.pow(2.0, bits-1)
    min_val = - bound * scale_factor
    max_val = (bound-1) * scale_factor
    input.clamp_(min_val, max_val)

class QuantizeFP:
    def __init__(self, dtype):
        if dtype == np.float32:
            self.ufixed_type = np.uint32
            self.fp_type = np.float32
            self.ctype = ctypes.c_uint
            self.and_value = "0x007FFFFF"
            self.or_value = "0x00800000"
            self.n_exponent = 8
            self.n_mantissa = 23
            self.n_bits = 32
            self.special_exponent = int("11111111",2)
        elif dtype == np.float64:
            self.ufixed_type = np.uint64
            self.fp_type = np.float64
            self.ctype = ctypes.c_ulong
            self.and_value = "0x000FFFFFFFFFFFFF"
            self.or_value =  "0x0010000000000000"
            self.n_exponent = 11
            self.n_mantissa = 52
            self.n_bits = 64
            self.special_exponent = int("11111111111",2)
        else:
            raise ValueError("Type not accepted for quantization (only float and double)")

def quantize_new(input, n_exponent_bits, n_mantissa_bits):
    q = QuantizeFP(input.dtype)

    XInt = input.ctypes.data_as(ctypes.POINTER(q.ctype * len(input)))

    new_np_array = np.ctypeslib.as_array(\
        (ctypes.c_uint * len(input)).from_address(\
            ctypes.addressof(XInt.contents)))
    
    and_array = np.full(shape=input.shape, 
                        fill_value=int(q.and_value, 16), 
                        dtype=q.ufixed_type)
    or_array = np.full(shape=input.shape, 
                       fill_value=int(q.or_value, 16), 
                       dtype=q.ufixed_type)

    mantissa = np.bitwise_or(np.bitwise_and(new_np_array, and_array), or_array)
    mantissa_shift = q.n_mantissa - n_mantissa_bits
    mantissa = np.left_shift(np.right_shift(mantissa, mantissa_shift), 
                                            mantissa_shift)
    mantissa_float = mantissa/2**(q.n_mantissa)

    bias = pow(2, q.n_exponent-1) - 1
    exponent_raw = np.right_shift(
                        np.left_shift(new_np_array, 1), (q.n_mantissa+1) )
    exponent = exponent_raw.astype(np.int) - bias
    
    # Clamp the exponent in the allowed range.
    quantized_exponent_max = pow(2, n_exponent_bits-1) - 1
    # (lose 1 to inf lose 1 to subnormal)
    quantized_exponent_min = - (pow(2, n_exponent_bits-1) - 2) 
    np.clip(exponent, quantized_exponent_min, quantized_exponent_max, out=exponent)

    exp_val = np.full(shape=input.shape, fill_value=2, dtype=q.fp_type)
    exp_val = np.power(exp_val, exponent)

    sign = np.right_shift(new_np_array, (q.n_bits-1) )
    sign_val = np.full(shape=input.shape, fill_value=-1, dtype=q.fp_type)
    sign_val.fill(-1)
    sign = np.power(sign_val, sign)

    # Given S, E, and M fields, an IEEE floating-point number has the value:
    #-1S × (1.0 + 0.M) × 2^E-bias
    reconstructed_val = (sign * mantissa_float * exp_val)

    # special numbers: e = 0 , means signifigand is subnormal.
    #       (−1)^signbit× 2^(min_exp) x 0.significandbits
    # e = 1111..., +inifinity when mantissa = 0, NaN when mantissa ne 0.  
    exponent_filter = np.logical_and( exponent_raw != 0, 
                                      exponent_raw != q.special_exponent)
    
    return np.where(exponent_filter, reconstructed_val, input)

def quantize_fp_(input, n_exponent_bits, n_mantissa_bits):
    in_shape = input.shape
    d_1 = input.reshape(input.numel())
    new_array = quantize_new(d_1.numpy(), n_exponent_bits, n_mantissa_bits)

    input.data = torch.tensor(new_array, 
                              dtype=input.dtype, 
                              requires_grad=input.requires_grad) \
                                .reshape(in_shape).data


# Monkey patch torch.Tensor
torch.Tensor.quantize_ = quantize_
torch.Tensor.saturate_ = saturate_

torch.Tensor.quantize_fp_ = quantize_fp_

#torch.cuda.FloatTensor.quantize_ = quantize_
#torch.cuda.FloatTensor.saturate_ = saturate_