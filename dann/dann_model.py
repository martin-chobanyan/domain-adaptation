"""Defines the DANN model (Domain Adversarial Neural Network)"""

from math import exp

import torch
import torch.nn as nn

from domain_adapt.nn.layers import ReverseGradient
from domain_adapt.nn.models import pretrained_alexnet_fc7
from domain_adapt.utils.misc import load_batch
from domain_adapt.utils.train import AverageKeeper, accuracy, softmax_pred


# ----------------------------------------------------------------------------------------------------------------------
# Models and layers
# ----------------------------------------------------------------------------------------------------------------------


class LabelClassifier(nn.Module):
    """Default label classifier architecture for the DANN model

    Parameters
    ----------
    num_classes: int
    input_dim: int, optional
    hidden_dim: int, optional
    """

    def __init__(self, num_classes, input_dim=4096, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DomainClassifier(nn.Module):
    """Default domain classifier architecture for the DANN model

    Parameters
    ----------
    input_dim: int, optional
    hidden_dim: int, optional
    """

    def __init__(self, input_dim=4096, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DANN(nn.Module):
    """Domain Adversarial Neural Network

    Parameters
    ----------
    feature_nn: nn.Module
        The feature extractor neural network
    label_nn: nn.Module
        The label classifier neural network
    domain_nn: nn.Module
        The domain classifier neural network
    """

    def __init__(self, feature_nn, label_nn, domain_nn):
        super().__init__()
        self.feature_nn = feature_nn
        self.label_nn = label_nn
        self.domain_nn = domain_nn
        self.reverse_grad = ReverseGradient()

    def get_label_classifier(self):
        """Concatenates the feature extractor and the label classifier

        This should be used when domain adaptation has been applied and the resulting model needs to be retrieved.

        Returns
        -------
        nn.Module
        """
        return nn.Sequential(self.feature_nn, self.label_nn)

    def forward(self, x_src, x_tgt, da_scale):
        """

        Parameters
        ----------
        x_src: torch.Tensor
            The source batch tensor
        x_tgt: torch.Tensor
            The target batch tensor
        da_scale: float, optional
            The scaling factor for the gradient reversal layer (default = 1)

        Returns
        -------
        tuple[torch.Tensor]
            A tuple containing the pre-softmax scores for the label classifier (only source) and the domain classifier
            (both source and target)
        """
        # get the source and target features
        src_features = self.feature_nn(x_src)
        tgt_features = self.feature_nn(x_tgt)

        # get the pre-softmax source label scores
        out_src_label = self.label_nn(src_features)

        # pass both source and target features through a gradient reversal layer
        src_features_reversed = self.reverse_grad(src_features, da_scale)
        tgt_features_reversed = self.reverse_grad(tgt_features, da_scale)

        # get the pre-softmax source and target domain scores
        out_src_domain = self.domain_nn(src_features_reversed)
        out_tgt_domain = self.domain_nn(tgt_features_reversed)

        return out_src_label, out_src_domain, out_tgt_domain


def load_model(num_classes):
    # set up the model components as specified in the paper
    feature_nn = pretrained_alexnet_fc7()
    label_nn = LabelClassifier(num_classes)
    domain_nn = DomainClassifier()
    return DANN(feature_nn, label_nn, domain_nn)


# ----------------------------------------------------------------------------------------------------------------------
# Training utilities for DANN
# ----------------------------------------------------------------------------------------------------------------------


def train_dann_epoch(model, src_loader, tgt_loader, criterion, optimizer, device, da_lambda):
    """Train a DANN model for one epoch

    Parameters
    ----------
    model: DANN
    src_loader: DataLoader
    tgt_loader: DataLoader
    criterion: callable
    optimizer: Optimizer
    device: torch.device
    da_scale: float
    """
    avg_acc = AverageKeeper()
    avg_label_loss = AverageKeeper()
    avg_domain_loss = AverageKeeper()
    model.train()
    for src_images, src_labels in src_loader:
        tgt_images, _ = load_batch(tgt_loader)

        src_images = src_images.to(device)
        src_labels = src_labels.to(device)
        tgt_images = tgt_images.to(device)

        # assign 1 to the source domain and 0 to the target domain
        num_src = len(src_images)
        num_tgt = len(tgt_images)
        src_domains = torch.ones(num_src, dtype=torch.int64).to(device)
        tgt_domains = torch.zeros(num_tgt, dtype=torch.int64).to(device)

        optimizer.zero_grad()
        out_src_label, out_src_domain, out_tgt_domain = model(src_images, tgt_images, da_lambda)
        loss_label = criterion(out_src_label, src_labels)

        # combine the source and target domain losses by weighted average
        loss_src_domain = criterion(out_src_domain, src_domains)
        loss_tgt_domain = criterion(out_tgt_domain, tgt_domains)
        loss_domain = (num_src * loss_src_domain + num_tgt * loss_tgt_domain) / (num_src + num_tgt)

        loss_total = loss_label + loss_domain
        loss_total.backward()
        optimizer.step()

        preds = softmax_pred(out_src_label.detach())
        avg_acc.add(accuracy(preds, src_labels))
        avg_label_loss.add(loss_label.detach().item())
        avg_domain_loss.add(loss_domain.detach().item())

    return avg_acc.calculate(), avg_label_loss.calculate(), avg_domain_loss.calculate()


class LRScheduler:
    """Learning rate scheduler

    This class defines the learning rate scheduler used in the DANN paper.

    Parameters
    ----------
    max_epochs: int
    init_lr: float
    alpha: float, optional
    beta: float, optional
    """

    def __init__(self, max_epochs, init_lr, alpha=10, beta=0.75):
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

     This hyperparameter controls scales the domain regularization loss.
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
