"""Defines the DANN model (Domain Adversarial Neural Network)"""

from math import exp

import torch
import torch.nn as nn
from torchvision.models import alexnet

from domain_adapt.nn import ReverseGradient


# ----------------------------------------------------------------------------------------------------------------------
# Training utilities for DANN
# ----------------------------------------------------------------------------------------------------------------------


def train_dann_epoch(model, src_loader, tgt_loader, criterion, optimizer, device, da_scale):
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
    model = model.train()
    for (src_images, src_labels), (tgt_images, _) in zip(src_loader, tgt_loader):
        src_images = src_images.to(device)
        src_labels = src_labels.to(device)
        tgt_images = tgt_images.to(device)

        n_src = len(src_images)
        src_domains = torch.ones(n_src, dtype=torch.int64).to(device)

        n_tgt = len(tgt_images)
        tgt_domains = torch.zeros(n_tgt, dtype=torch.int64).to(device)

        optimizer.zero_grad()
        out_src_label, out_src_domain, out_tgt_domain = model(src_images, tgt_images, da_scale)
        loss_src_label = criterion(out_src_label, src_labels)
        loss_src_domain = criterion(out_src_domain, src_domains)
        loss_tgt_domain = criterion(out_tgt_domain, tgt_domains)
        loss = loss_src_label + loss_src_domain + loss_tgt_domain
        loss.backward()
        optimizer.step()


# ----------------------------------------------------------------------------------------------------------------------
# Models and layers
# ----------------------------------------------------------------------------------------------------------------------


def pretrained_alexnet():
    """Prepares an AlexNet feature extractor pre-trained on ImageNet

    Note: The final fc8 layer is removed from the network, resulting in a 4096-dim output.

    Returns
    -------
    nn.Module
    """
    model = alexnet(pretrained=True)
    model.classifier = nn.Sequential(*[child for child in list(model.classifier.children())[:6]])
    return model


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

    def retrieve_model(self):
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
