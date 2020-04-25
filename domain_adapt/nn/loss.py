import torch


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
    mean_diff = (x1.sum(dim=0) / x1.size(0)) - (x2.sum(dim=0) / x2.size(0))
    return torch.norm(mean_diff, p=2).item()
