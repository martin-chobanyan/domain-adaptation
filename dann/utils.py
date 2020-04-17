from math import exp


# ----------------------------------------------------------------------------------------------------------------------
# Define the schedulers
# ----------------------------------------------------------------------------------------------------------------------

class LRScheduler:
    """Learning rate scheduler

    This class defines the learning rate scheduler used in the DANN paper.

    Parameters
    ----------
    max_epochs: int
    init_lr: float, optional
    alpha: float, optional
    beta: float, optional
    """

    def __init__(self, max_epochs, init_lr=0.01, alpha=10, beta=0.75):
        self.max_epochs = max_epochs
        self.lr_0 = init_lr
        self.alpha = alpha
        self.beta = beta

    def __call__(self, epoch):
        """Calculates the new learning rate given the epoch index

        Parameters
        ----------
        epoch: int
            The integer index of the current epoch

        Returns
        -------
        float
        """
        p = epoch / self.max_epochs
        lr_p = self.lr_0 / ((1 + self.alpha * p) ** self.beta)
        return lr_p


class DAScheduler:
    """The domain adaptation hyperparameter scheduler

     This hyperparamter controls scales the domain regularization loss.
     The value is initialized at zero and gradually changed to one.

     Parameters
     ----------
     max_epochs: int
     gamma: float
     """

    def __init__(self, max_epochs, gamma=10):
        self.max_epochs = max_epochs
        self.gamma = gamma

    def __call__(self, epoch):
        """Calculates the new domain adaptation parameter value given the epoch index

        Parameters
        ----------
        epoch: int

        Returns
        -------
        float
        """
        p = epoch / self.max_epochs
        lambda_p = (2 / (1 + exp(-self.gamma * p))) - 1
        return lambda_p
