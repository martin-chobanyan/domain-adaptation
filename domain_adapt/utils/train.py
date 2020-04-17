import torch
from .misc import accuracy, AverageKeeper, softmax_pred


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for an epoch and return the average training loss

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
    """Test the model for an epoch and return the average test loss

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
