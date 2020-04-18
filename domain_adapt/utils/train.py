from csv import writer as csv_writer

import torch
from .misc import AverageKeeper


class TrainingLogger:
    """A helper class to log the training status across different model runs

    Parameters
    ----------
    filepath: str
        The filepath for the output CSV log file
    """

    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'w') as file:
            header = ['Run', 'Epoch', 'Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy']
            writer = csv_writer(file)
            writer.writerow(header)

    def add_entry(self, run, epoch, train_loss, test_loss, train_acc, test_acc):
        with open(self.filepath, 'a') as file:
            writer = csv_writer(file)
            writer.writerow([run, epoch, train_loss, test_loss, train_acc, test_acc])


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
            acc_avg.add(accuracy(preds, labels.squeeze()))
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
