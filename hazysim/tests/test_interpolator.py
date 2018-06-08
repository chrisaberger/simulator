import pytest
import hazysim as sim
import torch

def test_sin():
    print("testing sin")

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

def main():
    test_sin()
    test_exp()

if __name__ == "__main__":
    main()