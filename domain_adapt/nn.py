import torch
from torch.nn import Module
from torch.autograd import Function


# ----------------------------------------------------------------------------------------------------------------------
# Useful domain adaptation layers
# ----------------------------------------------------------------------------------------------------------------------


class ReverseGradient(Module):
    """A gradient reversal layer

    This layer behaves like the identify function during the forward pass.
    During the backward pass, the sign of the incoming gradient to this layer is reversed and the value is scaled.
    """

    def __init__(self):
        super().__init__()
        self.reverse_grad = ReverseGradFunction()

    def forward(self, x, scale=1):
        return self.reverse_grad.apply(x, scale)


# ----------------------------------------------------------------------------------------------------------------------
# Custom autograd Functions
# ----------------------------------------------------------------------------------------------------------------------


class ReverseGradFunction(Function):
    """Defines a pytorch autograd Function which reverses the gradient during backpropagation"""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(torch.tensor(scale))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        return -scale * grad_output, None
