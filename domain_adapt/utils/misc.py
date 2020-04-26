"""Miscellaneous utility functions"""

import os

import torch


class AverageKeeper:
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


def create_dir(path):
    """Create the given directory (recursively) if it does not exist

    Parameters
    ----------
    path: str
    """
    if not os.path.exists(path) and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_script_dir(script_path):
    """Get the path of the directory containing a python script

    Examples
    --------
    >>> get_script_dir(__file__)  # this should be called from within the script to work

    Parameters
    ----------
    script_path: str

    Returns
    -------
    str
    """
    return os.path.split(os.path.realpath(script_path))[0]


def load_batch(loader):
    """Load a batch from a pytorch DataLoader

    Note: The `shuffle` arg in the DataLoader instance must be true in order to get a random batch during each call

    Parameters
    ----------
    loader: DataLoader

    Returns
    -------
    tuple
        The custom batch tuple from the DataLoader object
    """
    return next(iter(loader))

def get_device():
    """Get the cuda device if it is available

    Note: this assumes that there is only one GPU device

    Returns
    -------
    torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
