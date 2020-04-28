from csv import writer as csv_writer

import torch
from .misc import AverageKeeper


class TrainingLogger:
    """A helper class to log the training status across different model runs

    Parameters
    ----------
    filepath: str
        The filepath for the output CSV log file
    header: list[str], optional
        The custom header for the CSV file. If not specified, a default header will be initialized.
    """

    def __init__(self, filepath, header=None):
        self.filepath = filepath
        self.header = header
        if self.header is None:
            self.header = ['run', 'epoch', 'src_loss', 'tgt_loss', 'src_acc', 'tgt_acc']
        with open(self.filepath, 'w') as file:
            writer = csv_writer(file)
            writer.writerow(self.header)

    def add_entry(self, *args):
        """Add a row to the logger file
        Each value in the entry must align with the specified header.
        If no header was specified, then the inputs must be in the following order:
        1. run index
        2. epoch index
        3. source loss
        4. target loss
        5. source accuracy
        6. target accuracy
        """
        assert len(args) == len(self.header), "Number of inputs does not match the header"
        with open(self.filepath, 'a') as file:
            writer = csv_writer(file)
            writer.writerow([*args])


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
        The predicted labels as a 1D LongTensor
    targets: torch.LongTensor
        The ground truth labels as a 1D LongTensor

    Returns
    -------
    float
    """
    return (preds == targets).sum().item() / len(targets)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train a vanilla neural network for an epoch

    Parameters
    ----------
    model: torch.nn.Module
        The pytorch neural network model
    loader: torch.utils.data.DataLoader
        The dataloader that will shuffle and batch the dataset
    criterion: nn.Module/callable
        The loss criterion for the model
    optimizer: pytorch Optimizer
        The optimizer for this model
    device: torch.device
        The device for where the model will be trained

    Returns
    -------
    tuple[float]
        A tuple containing the average loss and accuracy across the epoch
    """
    loss_avg = AverageKeeper()
    acc_avg = AverageKeeper()
    model = model.train()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_avg.add(loss.detach().item())

        # isolate the predictions and calculate the accuracy
        preds = softmax_pred(out.detach())
        acc_avg.add(accuracy(preds, labels))
    return loss_avg.calculate(), acc_avg.calculate()


def test_epoch(model, loader, criterion, device):
    """Test a vanilla neural network for an epoch

    Parameters
    ----------
    model: torch.nn.Module
        The pytorch neural network model
    loader: torch.utils.data.DataLoader
        The dataloader that will shuffle and batch the dataset
    criterion: nn.Module/callable
        The loss criterion for the model
    device: torch.device
        The device for where the model will be trained

    Returns
    -------
    tuple[float]
        A tuple containing the average loss and accuracy across the epoch
    """
    loss_avg = AverageKeeper()
    acc_avg = AverageKeeper()
    model = model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            loss_avg.add(loss.detach().item())

            preds = softmax_pred(out.detach())
            acc_avg.add(accuracy(preds, labels))
    return loss_avg.calculate(), acc_avg.calculate()


def checkpoint(model, filepath):
    """Save the state of the model

    To restore the model do the following:
    >> the_model = TheModelClass(*args, **kwargs)
    >> the_model.load_state_dict(torch.load(PATH))

    Parameters
    ----------
    model: nn.Module
        The pytorch model to be saved
    filepath: str
        The filepath of the pickle
    """
    torch.save(model.state_dict(), filepath)
