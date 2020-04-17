"""Defines the DANN model (Domain Adversarial Neural Network)"""

import torch.nn as nn
from torchvision.models import alexnet


def pretrained_alexnet():
    model = alexnet(pretrained=True)
    model.classifier = nn.Sequential(*[child for child in list(model.classifier.children())[:6]])
    return model