import math
import torch
from torch.nn.parameter import Parameter
from .functional import *
from .base import Base

class Interpolator(Base):

    def __init__(self, fn, kind='linear'):
        super(Interpolator, self).__init__()
        self.fn = FInterpolator(fn, kind)

    def adapt(self, start, end, delta, hmin):
        print(self.fn)
        self.fn.adapt_linear(start, end, delta, hmin)

    def forward(self, input):
        result = self.fn.apply(input, self.fn)
        return result