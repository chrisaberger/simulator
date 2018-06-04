
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

def binRep(num):
    binNum = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
    logging.debug("bits: " + binNum.rjust(32,"0"))
    mantissa = "1" + binNum[-23:]
    logging.debug("sig (bin): " + mantissa.rjust(24))
    mantInt = int(mantissa,2)/2**23
    logging.debug("sig (float): " + str(mantInt))
    base = int(binNum[-31:-23],2)-127
    logging.debug("base:" + str(base))
    sign = 1-2*("1"==binNum[-32:-31].rjust(1,"0"))
    logging.debug("sign:" + str(sign))
    logging.debug("recreate:" + str(sign*mantInt*(2**base)))

def quantize_float(num, n_quantized_exp, n_quantized_mantissa):
    n_input_bits = None
    n_input_mantissa = None
    n_input_mantissa = None
    binary_number = None
    special_exponent = None

    if num.dtype == torch.float64:
        n_input_bits = 64
        n_input_mantissa = 52
        n_input_exponent = 11
        special_exponent = int("11111111111",2)
        # The [2:] strips off the '0b' in the returned string.
        binary_number = \
            bin(ctypes.c_ulong.from_buffer(ctypes.c_double(num)).value)[2:]
    elif num.dtype == torch.float32:
        n_input_bits = 32
        n_input_mantissa = 23
        n_input_exponent = 8
        special_exponent = int("11111111",2)
        # The [2:] strips off the '0b' in the returned string.
        binary_number = \
            bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
    else:
        raise ValueError("Type not accepted for quantization (only float and double)")

    # Mantissa is 1.****** 
    mantissa_binary = "1" + binary_number[-n_input_mantissa:]
    mantissa_binary.rjust(n_input_mantissa+1)

    # Use base 2 during cast.
    mantissa_int = int(mantissa_binary, 2)

    # Chop off the mantissa bits that no longer fit.
    mantissa_int = ( (mantissa_int >> (n_input_mantissa-n_quantized_mantissa)) \
                         << (n_input_mantissa-n_quantized_mantissa) )
    mantissa_float = mantissa_int/2**n_input_mantissa

    exponent = int(binary_number[-(n_input_bits-1):-n_input_mantissa], 2)

    # special numbers: e = 0 , means signifigand is subnormal.
    #       (−1)^signbit× 2^(min_exp) x 0.significandbits
    # e = 1111..., +inifinity when mantissa = 0, NaN when mantissa ne 0.
    if exponent != 0 and exponent != special_exponent:
        bias = pow(2, n_input_exponent-1) - 1
        exponent = exponent - bias
        # Clamp the exponent in the allowed range.
        quantized_exponent_max = pow(2, n_quantized_exp-1) - 1
        # (lose 1 to inf lose 1 to subnormal)
        quantized_exponent_min = - (pow(2, n_quantized_exp-1) - 2) 
        exponent = quantized_exponent_max if exponent > quantized_exponent_max \
                                          else exponent 
        exponent = quantized_exponent_min if exponent < quantized_exponent_min \
                                          else exponent
    elif exponent == special_exponent:
        # if you get Nan or Inf just return it.
        return num

    sign = 1-2*("1"==binary_number[-n_input_bits:-(n_input_bits-1)] \
                .rjust(1,"0"))

    print("Mantissa: " + str(mantissa_float))
    print("Exponent: " + str(exponent))
    print("Sign: " + str(sign))

    #logging.debug("Mantissa: " + str(mantissa_float))
    #logging.debug("Exponent: " + str(exponent))
    #logging.debug("Sign: " + str(sign))

    return sign * mantissa_float * (2**exponent)

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
    exponent_raw = np.right_shift(np.left_shift(new_np_array, 1), (q.n_mantissa+1) )
    exponent = exponent_raw.astype(np.int) - bias
    
    # Clamp the exponent in the allowed range.
    quantized_exponent_max = pow(2, n_exponent_bits-1) - 1
    exponent[exponent > quantized_exponent_max] = quantized_exponent_max
    # (lose 1 to inf lose 1 to subnormal)
    quantized_exponent_min = - (pow(2, n_exponent_bits-1) - 2) 
    exponent[exponent < quantized_exponent_min] = quantized_exponent_min

    exp_val = np.full(shape=input.shape, fill_value=2, dtype=q.fp_type)
    exp_val = np.power(exp_val, exponent)

    sign = np.right_shift(new_np_array, (q.n_bits-1) )
    sign_val = np.full(shape=input.shape, fill_value=-1, dtype=q.fp_type)
    sign_val.fill(-1)
    sign = np.power(sign_val, sign)

    #print(sign)
    #print(mantissa_float)
    #print(exp_val)

    # special numbers: e = 0 , means signifigand is subnormal.
    #       (−1)^signbit× 2^(min_exp) x 0.significandbits
    # e = 1111..., +inifinity when mantissa = 0, NaN when mantissa ne 0.
    reconstructed_val = (sign * mantissa_float * exp_val)
  
    exponent_filter = np.logical_and(exponent_raw != 0, exponent_raw != q.special_exponent)
    return np.where(exponent_filter, reconstructed_val, input)

def quantize_fp_(input, n_exponent_bits, n_mantissa_bits):
    # Given S, E, and M fields, an IEEE floating-point number has the value:
    #-1S × (1.0 + 0.M) × 2^E-bias
    # exponent can be in range +2^(E-1)-1
    # to -(2^(E-1)-1-1) (lose 1 to inf lose 1 to subnormal)
    # 
    # special numbers: e = 0 , means signifigand is subnormal.
    #       (−1)^signbit× 2^(min_exp) x 0.significandbits
    # 
    # e = 1111..., +inifinity when mantissa = 0, NaN when mantissa ne 0.

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