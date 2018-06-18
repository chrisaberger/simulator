import math
import torch
import struct
import ctypes
import sys
import logging
import numpy as np

class Quantizer:
    n_exponent_bits = 8
    n_mantissa_bits = 23

    n_bits = 32
    scale_factor = 1.0
    quantize_fixed = False

    @staticmethod
    def set_float_precision(num_bits, num_mantissa_bits):
        Quantizer.n_exponent_bits = num_bits - (num_mantissa_bits+1)
        Quantizer.n_mantissa_bits = num_mantissa_bits
        Quantizer.quantize_fixed = False

    @staticmethod
    def set_fixed_precision(scale_factor, num_bits):
        Quantizer.n_bits = num_bits
        Quantizer.scale_factor = scale_factor
        Quantizer.quantize_fixed = True

def quantize_(input):
    if Quantizer.quantize_fixed:
        input.quantize_fixed_()
    else:
        input.quantize_float_()

def quantize_fixed_(input, scale_factor = None, bits = None, biased=False):
    inp = input.clone()
    if bits == None:
        bits = Quantizer.n_bits
    if scale_factor == None:
        scale_factor = Quantizer.scale_factor

    assert bits >= 1, bits
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    if biased:
        adj_val = 0.5
    else:
        # Generate tensor of random values from [0,1]
        adj_val = torch.Tensor(inp.size()).type(inp.type()).uniform_()

    rounded = inp.div_(scale_factor).add_(adj_val).floor_()
    clipped_value = rounded.clamp_(min_val, max_val)
    clipped_value *= scale_factor
    input.data.copy_(inp.data)

class IEEEFloatingPointData:
    def __init__(self, dtype):
        if dtype == np.float32:
            self.ufixed_type = np.uint32
            self.fp_type = np.float32
            self.ctype = ctypes.c_uint
            self.mantissa_and_value = "0x007FFFFF"
            self.mantissa_or_value = "0x00800000"
            self.exp_and_value =     "0x7FFFFFFF"
            self.n_exponent = 8
            self.n_mantissa = 23
            self.n_bits = 32
            self.special_exponent = int("11111111",2)
        elif dtype == np.float64:
            self.ufixed_type = np.uint64
            self.fp_type = np.float64
            self.ctype = ctypes.c_ulong
            self.mantissa_and_value = "0x000FFFFFFFFFFFFF"
            self.mantissa_or_value =  "0x0010000000000000"
            self.exp_and_value =      "0x7FFFFFFFFFFFFFFF"
            self.n_exponent = 11
            self.n_mantissa = 52
            self.n_bits = 64
            self.special_exponent = int("11111111111",2)
        else:
            raise ValueError("Type "+str(dtype)+" not accepted for quantization"
                             "(only float and double).")

def break_down_fp_(input):
    in_dtype = input.dtype
    in_shape = input.shape
    d_1 = input.reshape(input.numel())
    d_1 = d_1.detach().numpy()

    q = IEEEFloatingPointData(d_1.dtype)

    d_1 = np.ascontiguousarray(d_1)
    XInt = d_1.ctypes.data_as(ctypes.POINTER(q.ctype*len(d_1)))

    new_np_array = np.ctypeslib.as_array(\
        (q.ctype * len(d_1)).from_address(\
            ctypes.addressof(XInt.contents)))

    ################## Process Exponent. ##################
    bias = pow(2, q.n_exponent-1) - 1
    exponent_raw = np.right_shift(
                        np.left_shift(new_np_array, 1), (q.n_mantissa+1) )
    subnormal_nums = exponent_raw == 0
    exponent_raw[exponent_raw == 0] = 1 # subnormal numbers are 2^(-bias-1)
    exponent = exponent_raw.astype(np.int) - bias

    ################## Process Mantissa. ##################
    mantissa_and_array = np.full(shape=d_1.shape, 
                        fill_value=int(q.mantissa_and_value, 16), 
                        dtype=q.ufixed_type)
    mantissa_or_array = np.full(shape=d_1.shape, 
                       fill_value=int(q.mantissa_or_value, 16), 
                       dtype=q.ufixed_type)
    mantissa_subnormal_or_array = np.full(shape=d_1.shape, 
                       fill_value=int(0), 
                       dtype=q.ufixed_type)
    # Do not add 1.<mantissa> to subnormal numbers (ie take 'mantissa_or_array')
    or_stuff = np.where(subnormal_nums, mantissa_subnormal_or_array, mantissa_or_array)
    mantissa = np.bitwise_or(np.bitwise_and(new_np_array, mantissa_and_array), 
                             or_stuff)
    mantissa_float = mantissa/2**(q.n_mantissa)

    ################## Process Sign. ##################
    sign = np.right_shift(new_np_array, (q.n_bits-1) )
    sign_val = np.full(shape=d_1.shape, fill_value=-1, dtype=q.fp_type)
    sign_val.fill(-1)
    sign = np.power(sign_val, sign)

    return (torch.tensor(sign, dtype=in_dtype).reshape(in_shape), 
            torch.tensor(exponent, dtype=in_dtype).reshape(in_shape), 
            torch.tensor(mantissa_float, dtype=in_dtype).reshape(in_shape))

def quantize_floating_point_(input, n_exponent_bits, n_mantissa_bits):
    q = IEEEFloatingPointData(input.dtype)

    input = np.ascontiguousarray(input)
    XInt = input.ctypes.data_as(ctypes.POINTER(q.ctype*len(input)))

    new_np_array = np.ctypeslib.as_array(\
        (q.ctype * len(input)).from_address(\
            ctypes.addressof(XInt.contents)))

    ################## Process Exponent. ##################
    bias = pow(2, q.n_exponent-1) - 1
    exponent_raw = np.right_shift(
                        np.left_shift(new_np_array, 1), (q.n_mantissa+1) )
    subnormal_nums = exponent_raw == 0
    exponent_raw[exponent_raw == 0] = 1 # subnormal numbers are 2^(-bias-1)
    exponent = exponent_raw.astype(np.int) - bias
    # Clamp the exponent in the allowed range.
    quantized_exponent_max = pow(2, n_exponent_bits-1) - 1
    # (lose 1 to inf lose 1 to subnormal)
    quantized_exponent_min = - (pow(2, n_exponent_bits-1) - 2) 
    np.clip(exponent, 
            quantized_exponent_min, 
            quantized_exponent_max, 
            out=exponent)
    exp_val = np.full(shape=input.shape, fill_value=2, dtype=q.fp_type)
    exp_val = np.power(exp_val, exponent)

    ################## Process Mantissa. ##################
    mantissa_and_array = np.full(shape=input.shape, 
                        fill_value=int(q.mantissa_and_value, 16), 
                        dtype=q.ufixed_type)
    mantissa_or_array = np.full(shape=input.shape, 
                       fill_value=int(q.mantissa_or_value, 16), 
                       dtype=q.ufixed_type)
    mantissa_subnormal_or_array = np.full(shape=input.shape, 
                       fill_value=int(0), 
                       dtype=q.ufixed_type)
    # Do not add 1.<mantissa> to subnormal numbers (ie take 'mantissa_or_array')
    or_stuff = np.where(subnormal_nums, mantissa_subnormal_or_array, mantissa_or_array)
    mantissa = np.bitwise_or(np.bitwise_and(new_np_array, mantissa_and_array), 
                             or_stuff)
    mantissa_shift = q.n_mantissa - n_mantissa_bits
    mantissa = np.left_shift(np.right_shift(mantissa, mantissa_shift), 
                                            mantissa_shift)
    mantissa_float = mantissa/2**(q.n_mantissa)

    ################## Process Sign. ##################
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
    exponent_filter = exponent_raw != q.special_exponent
    return np.where(exponent_filter, reconstructed_val, input)

def quantize_float_(input, n_exponent_bits = None, n_mantissa_bits = None):
    if n_exponent_bits is None:
        n_exponent_bits = Quantizer.n_exponent_bits
    if n_mantissa_bits is None:
        n_mantissa_bits = Quantizer.n_mantissa_bits

    in_shape = input.shape
    d_1 = input.reshape(input.numel()).detach()
    new_array = quantize_floating_point_(
                    d_1.numpy(), n_exponent_bits, n_mantissa_bits)
    input.data.copy_(torch.tensor(new_array, 
                              dtype=input.dtype, 
                              requires_grad=input.requires_grad) \
                                .reshape(in_shape).data)


# Monkey patch torch.Tensor
torch.Tensor.quantize_ = quantize_
torch.Tensor.quantize_float_ = quantize_float_
torch.Tensor.quantize_fixed_ = quantize_fixed_
torch.Tensor.break_down_fp_ = break_down_fp_

