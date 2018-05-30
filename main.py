import interpolator as interp
import torch.nn.functional as F
import torch

a_interp = torch.tensor([1.0,0.0,-1.0,2.0],requires_grad=True)
a_actual = torch.tensor([1.0,0.0,-1.0,2.0],requires_grad=True)

delta = 0.01
hmin = 0.01
sin = torch.sin

isin = interp.Interpolator(sin)
isin.adapt(0, 10, delta, hmin)
isin.plot()
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
