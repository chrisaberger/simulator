import pytest
import hazysim as sim
import torch

def test_sin():
    a_interp = torch.tensor([1.0,0.0,-1.0,2.0], requires_grad=True)
    a_actual = torch.tensor([1.0,0.0,-1.0,2.0], requires_grad=True)

    sin = torch.sin
    isin = sim.nn.Interpolator(torch.sin)

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

    print(torch.abs(actual - sin_result).sum().item())
    assert(torch.abs(actual - sin_result).sum().item() <= 0.0051)
    print(torch.abs(a_actual.grad - a_interp.grad).sum().item())
    assert(torch.abs(a_actual.grad - a_interp.grad).sum().item() <= 0.0065)

def test_exp():
    a_interp = torch.tensor([-90.0,-10,10,75], requires_grad=True)
    a_actual = torch.tensor([-90.0,-10,10,75], requires_grad=True)

    iexp = sim.nn.Interpolator(torch.exp)
    iexp.chunk(min = -100, max = 100, num_points = 1000)

    exp_result = iexp(a_interp)
    exp_result.sum().backward()
    print("Forward Interpolator")
    print(exp_result)
    print("Backward Interpolator")
    print(a_interp.grad)

    print("Forward Acutal")
    actual = torch.exp(a_actual)

    print(actual)
    print("Backward Actual")
    actual.sum().backward()
    print(a_actual.grad)

    assert(torch.abs(actual - exp_result).sum().item() == 0.0)
    assert(torch.abs(a_actual.grad - a_interp.grad).sum().item() == 0.0)

def test_log():
    a_interp = torch.tensor([0.05,0.5,1.0,5.0],requires_grad=True)
    a_actual = torch.tensor(a_interp.data, requires_grad =True)

    log_fn = sim.nn.LogInterpolator()

    print("Forward")
    log_interp = log_fn.forward(a_interp)
    log_interp.sum().backward()
    print(log_interp)
    print(a_interp.grad)

    log_actual = torch.log(a_actual)
    log_actual.sum().backward()
    print(log_actual)
    print(a_actual.grad)

    print(torch.abs(log_actual - log_interp).sum().item())
    assert(torch.abs(log_actual - log_interp).sum().item() <= 0.003)
    assert(torch.abs(a_actual.grad - a_interp.grad).sum().item() == 0.0)

def main():
    test_sin()
    test_exp()
    test_log()

if __name__ == "__main__":
    main()