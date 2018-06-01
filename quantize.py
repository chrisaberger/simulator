
import math
import torch
import struct
import ctypes
import sys
import logging

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

    sign = 1-2*("1"==binary_number[-n_input_bits:-(n_input_bits-1)]\
                .rjust(1,"0"))

    #logging.debug("Mantissa: " + str(mantissa_float))
    #logging.debug("Exponent: " + str(exponent))
    #logging.debug("Sign: " + str(sign))

    return sign * mantissa_float * (2**exponent)

def quantize_fp_(input, n_exponent_bits, n_mantissa_bits):
    # Therefore, given S, E, and M fields, an IEEE floating-point number has the value:
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
    new_tensor = [quantize_float(i, n_exponent_bits, n_mantissa_bits) for i in d_1]
    input.data = torch.tensor(new_tensor, dtype=input.dtype).reshape(in_shape).data


# Monkey patch torch.Tensor
torch.Tensor.quantize_ = quantize_
torch.Tensor.saturate_ = saturate_

torch.Tensor.quantize_fp_ = quantize_fp_
torch.Tensor.saturate_ = saturate_

#torch.cuda.FloatTensor.quantize_ = quantize_
#torch.cuda.FloatTensor.saturate_ = saturate_