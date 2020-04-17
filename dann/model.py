"""Defines the DANN model (Domain Adversarial Neural Network)"""

import torch.nn as nn
from torchvision.models import alexnet

from domain_adapt.nn import ReverseGradient


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
    reverse_scale: float, optional
        The scaling factor for the gradient reversal layer (default = 1)
    """

    def __init__(self, feature_nn=None, label_nn=None, domain_nn=None, reverse_scale=1):
        super().__init__()
        self.feature_nn = feature_nn
        self.label_nn = label_nn
        self.domain_nn = domain_nn
        self.reverse_grad = ReverseGradient(scale=reverse_scale)

    def retrieve_model(self):
        """Concatenates the feature extractor and the label classifier

        This should be used when domain adaptation has been applied and the resulting model needs to be retrieved.

        Returns
        -------
        nn.Module
        """
        return nn.Sequential(self.feature_nn, self.label_nn)

    def forward(self, x_src, x_tgt):
        # get the base source and target features
        src_features = self.feature_nn(x_src)
        tgt_features = self.feature_nn(x_tgt)

        # get the pre-softmax source label scores
        out_src_label = self.label_nn(src_features)

        # pass both source and target features through a gradient reversal layer
        src_features_ = self.reverse_grad(src_features)
        tgt_features_ = self.reverse_grad(tgt_features)

        # get the pre-softmax source and target domain scores
        out_src_domain = self.domain_nn(src_features_)
        out_tgt_domain = self.domain_nn(tgt_features_)

        return out_src_label, out_src_domain, out_tgt_domain
