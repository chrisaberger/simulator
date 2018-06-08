import pytest
import hazysim
import torch

def test_quantize_float_simple():
    a = torch.randn(5000,5000).float()    
    acutal_sum = a.sum()
    a.quantize_(n_exponent_bits=8, n_mantissa_bits=23)
    quantize_sum = a.sum()
    assert(acutal_sum == quantize_sum)

def test_quantize_double_simple():
    a = torch.randn(5000,5000).double()    
    acutal_sum = a.sum()
    a.quantize_(n_exponent_bits=8, n_mantissa_bits=23)
    quantize_sum = a.sum()
    assert(acutal_sum == quantize_sum)

def test_quantize_float():
    a = torch.randn(5000,5000).float()    
    acutal_sum = a.sum()
    a.quantize_(n_exponent_bits=5, n_mantissa_bits=4)
    quantize_sum = a.sum()
    assert(acutal_sum == quantize_sum)