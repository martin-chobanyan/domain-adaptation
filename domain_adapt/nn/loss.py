import torch
from torch.nn import Module


# ----------------------------------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------------------------------

class MaxMeanDiscrepancy(Module):
    def forward(self, x1, x2):
        return mmd(x1, x2)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def mmd(x1, x2):
    """Maximum Mean Discrepancy

    This function calculates the MMD between examples from two separate feature distributions.

    Parameters
    ----------
    x1: torch.Tensor
        Examples from a feature space with shape (num_examples, feature_dim)
    x2: torch.Tensor
        Examples from a feature space with shape (num_examples, feature_dim)

    Returns
    -------
    float
    """
    return torch.norm(x1.mean(dim=0) - x2.mean(dim=0), p=2)
