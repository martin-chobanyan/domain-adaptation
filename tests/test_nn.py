from copy import deepcopy

import torch
from torch.nn import Linear
from domain_adapt.nn import ReverseGradient


def test_gradient_reversal():
    for scale in [0, 1, 2]:
        x = torch.rand((1, 2))
        fc1 = Linear(2, 2)
        fc2 = Linear(2, 2)
        reverse_grad = ReverseGradient()

        # get the gradients without reversal
        y = fc2(fc1(x)).sum()
        y.backward()
        grad1 = deepcopy(fc1.weight.grad.data)

        # get the gradients with reversal
        fc1.weight.grad.data.zero_()
        y = fc2(reverse_grad(fc1(x), scale=scale)).sum()
        y.backward()
        grad2 = fc1.weight.grad.data

        # ensure the gradient sign is flipped and scaled
        assert torch.all(torch.eq(-scale * grad1, grad2)).item(), f"ReverseGradient does not work for scale={scale}"
