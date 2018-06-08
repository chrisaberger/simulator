import math
import torch
from torch.nn.parameter import Parameter
from .functional import *
from .base import Base

class Interpolator(Base):
    def __init__(self, fn, kind='linear'):
        super(Interpolator, self).__init__()
        self.fn = FInterpolator(fn, kind)

    def chunk(self, min, max, num_points):
        self.fn.naive_chunking(min, max, num_points)

    def adapt(self, start, end, delta, hmin):
        self.fn.adapt_linear(start, end, delta, hmin)

    def forward(self, input):
        result = self.fn.apply(input, self.fn)
        return result.float()

class LogInterpolator(Base):
    def __init__(self, kind='linear'):
        super(LogInterpolator, self).__init__()
        fn = FInterpolator(torch.log2, kind)
        mantissa_min = 1.19e-7 # For float32 
        mantissa_max = 2.0 # For float 32
        fn.adapt_linear(mantissa_min, mantissa_max, 1e-4, 1e-4)
        #fn.naive_chunking(mantissa_min, mantissa_max, 100)
        self.interp = fn.forward_fn_interpolate

    def forward(self, input):
        return FLog.apply(input, self.interp)
        