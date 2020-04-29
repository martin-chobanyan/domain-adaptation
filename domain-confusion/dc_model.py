import torch.nn as nn
from tqdm import tqdm

from domain_adapt.nn.loss import mmd
from domain_adapt.nn.models import pretrained_alexnet_fc7
from domain_adapt.utils.misc import load_batch
from domain_adapt.utils.train import accuracy, AverageKeeper, softmax_pred


class DomainConfusionNN(nn.Module):
    """Deep Domain Confusion CNN

    Parameters
    ----------
    base_nn: nn.Module
        The pre-trained base CNN feature extractor for the model
    base_width: int
        The dimension of the flattened output of base_nn
    adapt_width: int
        The dimension of the adaptation layer
    num_classes: int
        The number of classes for the classification layer
    """

    def __init__(self, base_nn, base_width, adapt_width, num_classes):
        super().__init__()
        self.base_nn = base_nn
        self.adapt_nn = nn.Linear(base_width, adapt_width)
        self.classify_nn = nn.Linear(adapt_width, num_classes)

    def strip_classifier(self):
        """Retrieve the network without the final classifier layer"""
        return nn.Sequential(self.base_nn, self.adapt_nn)

    def invariant_features(self, x):
        """Get the output of the adaptation layer"""
        return self.adapt_nn(self.base_nn(x))

    def forward(self, x):
        """Get the classification scores (for normal use once trained)"""
        return self.classify_nn(self.invariant_features(x))


def load_model(width=256):
    """Load the default version of the domain confusion model

    The default version uses a pre-trained AlexNet as the backbone excluding the final fc8 layer,
    an adaptation width of 256, and 31 classes (for Office-31).
    """
    return DomainConfusionNN(pretrained_alexnet_fc7(), base_width=4096, adapt_width=width, num_classes=31)


def calculate_mmd(model, src_loader, tgt_loader, device, progress=True):
    """Calculate Maximum Mean Discrepancy across two sets of data

    This function is used to for choosing the position and width of the adaptation layer.

    Parameters
    ----------
    model: nn.Module
    src_loader: DataLoader
    tgt_loader: DataLoader
    device: torch.device
    progress: bool, optional

    Returns
    -------
    float
    """
    model = model.eval()
    model = model.to(device)

    mmd_values = []
    for src_imgs, _ in tqdm(src_loader, disable=(not progress)):
        tgt_imgs, _ = load_batch(tgt_loader)

        src_imgs = src_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)

        src_features = model(src_imgs)
        tgt_features = model(tgt_imgs)

        val = mmd(src_features, tgt_features)
        mmd_values.append(val)
    return sum(mmd_values) / len(mmd_values)


def train_domain_confusion(model, src_loader, tgt_loader, criterion, optimizer, device, da_lambda):
    """A single epoch of domain confusion training

    Note: The `shuffle` arg for the target DataLoader must be True in order to get a random batch for each iteration.

    Parameters
    ----------
    model: DomainConfusionNN
    src_loader: DataLoader
    tgt_loader: DataLoader
    criterion: callable
    optimizer: Optimizer
    device: torch.device
    da_lambda: float

    Returns
    -------
    tuple[float]
        The training source cross entropy loss and accuracy
    """
    acc_avg = AverageKeeper()
    loss_avg = AverageKeeper()
    model = model.train()
    for src_images, src_labels in src_loader:
        tgt_images, _ = load_batch(tgt_loader)

        src_images = src_images.to(device)
        src_labels = src_labels.to(device)
        tgt_images = tgt_images.to(device)

        optimizer.zero_grad()

        # collect domain invariant features from the adapt bottleneck layer
        src_features = model.invariant_features(src_images)
        tgt_features = model.invariant_features(tgt_images)

        # calculate the source label loss
        label_scores = model.classify_nn(src_features)
        loss_label = criterion(label_scores, src_labels)

        # calculate the domain confusion loss
        loss_mmd = mmd(src_features, tgt_features)

        loss_total = loss_label + da_lambda * (loss_mmd ** 2)
        loss_total.backward()
        optimizer.step()

        preds = softmax_pred(label_scores.detach())
        acc_avg.add(accuracy(preds, src_labels))
        loss_avg.add(loss_label.detach().item())
    return loss_avg.calculate(), acc_avg.calculate()
