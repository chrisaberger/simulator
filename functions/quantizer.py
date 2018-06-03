import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensor.quantize import * 

class Quantizer(torch.autograd.Function):

    def __init__(self, fn, num_bits, kind = "fp"):
        """
        Initializes the function we will quantize around.
        """
        self.fn = fn
        self.num_bits = num_bits
        self.kind = kind

        a = torch.tensor([1.0,1,1,1,1,1,1,1,1,1], requires_grad=True)
        b = self.fn(a)
        b.sum().backward(retain_graph=True)
        print(a.grad_fn)
        self.grad_fn = b.grad_fn

    def grab_backward(self):
        print("HERE YE YE")
        b = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)
        a = torch.sin(b)
        a.sum().backward()
        print(a)
        print(a.requires_grad)
        print("HERE YE YE")

    @staticmethod
    def forward(ctx, input, self):
        forward_val = self.fn(input)
        print(input)
        backward_val = self.grad_fn(torch.tensor([2.0,2,1,1,1,1,1,1,1,1]))
        print("BACKWARD VAL")
        
        a = torch.tensor(input, requires_grad=True)
        a.allow_unreachable = False
        b = torch.sin(a)
        b.requires_grad = True
        b.allow_unreachable = False
        c = b.sum()
        c.requires_grad = True
        c.backward()

        print(a.grad)
        print(b.grad)
        print(c.grad)
        ctx.save_for_backward(input, a.grad)
        return self.fn(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, grad_output  = ctx.saved_tensors
        print("BACKWARD")
        print(grad_output)
        return grad_output, None
