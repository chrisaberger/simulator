import torch.nn.functional as F
import torch
import logging
from datetime import datetime
import os
import time

import hazysim.nn as sim


def test_exp():

    a_interp = torch.tensor([-90.0,-10,10,75], requires_grad=True)
    a_actual = torch.tensor([-90.0,-10,10,75], requires_grad=True)

    iexp = sim.Interpolator(torch.exp)
    iexp.chunk(min = -100, max = 100, num_points = 100)

    exp_result = iexp(a_interp)
    exp_result.sum().backward()
    print(exp_result)
    print(a_interp.grad)

    print("Forward Acutal")
    actual = torch.exp(a_actual)
    print(actual)
    print("Backward Actual")
    actual.sum().backward()
    print(a_actual.grad)

test_exp()
exit()

def test_interpolate():
    a_interp = torch.tensor([1.0,0.0,-1.0,2.0], requires_grad=True)
    a_actual = torch.tensor([1.0,0.0,-1.0,2.0], requires_grad=True)

    sin = torch.sin
    isin = sim.Interpolator(torch.sin)

    """
    Grid the points using an adaptive method.
    Will adapt points in the range [start, end].
    Each subinterval [x(i),x(i+1)] is either <= hmin in length or has the property 
    that at its midpoint m, |f(m) - L(m)| <= delta where L(x) is the line that 
    connects (x(i),y(i)) and (x(i+1),y(i+1)).
    """
    isin.adapt(start = -8, end = 10, delta = 0.01, hmin = 0.01)
    isin.chunk(min = -8, max = 10, num_points = 100)
    print(isin)
    sin_result = isin(a_interp)
    sin_result.sum().backward()
    print(sin_result)
    print(a_interp.grad)

    print("Forward Acutal")
    actual = torch.sin(a_actual)
    print(actual)
    print("Backward Actual")
    actual.sum().backward()
    print(a_actual.grad)

    exit()

    """
    Plot the forwards interpolation first, then the backwards interpolation.
    """
    isin.plot()

    # Forwards pass.

    print("Forward Interp")
    print(sin_fn)
    print("Forward Acutal")
    actual = torch.sin(a_actual)
    print(actual)

    sin_fn.sum().backward()
    print("Backward Interp")
    print(a_interp.grad)

    print("Backward Actual")
    actual.sum().backward()
    print(a_actual.grad)


test_interpolate()
exit()

def test_quantize():
    a = torch.randn(5000,5000)
    
    print(a)
    b = a.clone()

    t0 = time.time()
    a.quantize_(n_exponent_bits=4, n_mantissa_bits=5)
    #a = a*a
    print(len(a))
    t1 = time.time()
    print("Quantize FP: " + str(t1-t0))
    print(a)
    
    """
    t0 = time.time()
    b.quantize_(1, 11)
    #b = b*b
    print(len(b))
    t1 = time.time()
    print("Quantize Old: " + str(t1-t0))
    print(b)
    """

    filename = str(datetime.now().strftime('%Hh%Mm%Ss_%m-%d-%Y'))+".log"
    logging.basicConfig(filename="log/"+filename,level=logging.DEBUG)

    print("HERE")
    fn = sim.Linear(10, 10)
    fn2 = sim.Linear(10, 5)

    def hookFunc(module, gradInput, gradOutput):
        print("YOU KNOW")
    fn2.register_backward_hook(hookFunc) 

    fn.register_precision(10, 5)
    fn.register_precision(11, 5)
    fn.register_precision(20, 1)

    fn2.register_precision(23, 2)


    #fn2.register_precision(10, 5)
    print(fn._backward_hooks)
    print(fn2._backward_hooks)

    x = torch.randn(10,10)
    out = fn2(fn(x))
    out.sum().backward()

if not os.path.exists("log"):
    os.makedirs("log")

test_quantize()

#print(w.grad)