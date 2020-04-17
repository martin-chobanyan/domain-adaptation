"""Miscellaneous utility functions"""

import os

import torch


class AverageKeeper(object):
    """
    Helper class to keep track of averages
    """

    def __init__(self):
        self.sum = 0
        self.n = 0
        self.running_avg = []

    def add(self, x):
        """Update the current running sum"""
        self.sum += x
        self.n += 1

    def calculate(self):
        """Calculate the current average and append to the running average"""
        avg = self.sum / self.n if self.n != 0 else 0
        self.running_avg.append(avg)
        return avg

    def reset(self, complete=False):
        """Reset the average counter

        Parameters
        ----------
        complete: bool
            If complete is True, then the running average will be reset as well
        """
        self.sum = 0
        self.n = 0
        if complete:
            self.running_avg = []


def softmax_pred(linear_out):
    """Apply softmax and collect the predictions

    Parameters
    ----------
    linear_out: torch.Tensor
        The tensor output of the pytorch nn model. Assumes 2D, stacked vectors

    Returns
    -------
    torch.LongTensor
        A tensor of the argmax for each vector
    """
    softmax_out = torch.softmax(linear_out, dim=1)
    pred = torch.argmax(softmax_out, dim=1)
    return pred


def accuracy(preds, targets):
    """Calculate the accuracy

    Parameters
    ----------
    preds: torch.LongTensor
    targets: torch.LongTensor

    Returns
    -------
    float
    """
    return (preds == targets).sum().item() / len(targets)


def create_dir(path):
    """Create the given directory (recursively) if it does not exist

    Parameters
    ----------
    path: str
    """
    if not os.path.exists(path) and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
