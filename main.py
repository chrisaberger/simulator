import interpolator as interp
import torch.nn.functional as F
import torch

a_interp = torch.tensor([1.0,0.0,-1.0,2.0], requires_grad=True)
a_actual = torch.tensor([1.0,0.0,-1.0,2.0], requires_grad=True)

sin = torch.sin
isin = interp.Interpolator(sin)

"""
Grid the points using an adaptive method.

Will adapt points in the range [start, end].

Each subinterval [x(i),x(i+1)] is either <= hmin in length or has the property 
that at its midpoint m, |f(m) - L(m)| <= delta where L(x) is the line that 
connects (x(i),y(i)) and (x(i+1),y(i+1)).
"""
isin.adapt(start=0, end=10, delta=0.01, hmin=0.01)

"""
Plot the forwards interpolation first, then the backwards interpolation.
"""
isin.plot()

# Forwards pass.
sin_fn = isin.apply(a_interp, isin)

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
