"""This script searches for the best width of the adaptation layer attached to fc7 in AlexNet """
import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from domain_adapt.data import Office31
from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.nn.models import pretrained_alexnet_fc7
from domain_adapt.utils.misc import get_device
from domain_adapt.utils.train import TrainingLogger, train_epoch, test_epoch

from layer_search import calculate_mmd

BATCH_SIZE = 32
NUM_CLASSES = 31


class DomainConfusion(nn.Module):
    def __init__(self, base_nn, base_width, adapt_width, num_classes):
        super().__init__()
        self.base_nn = base_nn
        self.adapt_nn = nn.Sequential(nn.Linear(base_width, adapt_width), nn.ReLU())
        self.classifier = nn.Linear(adapt_width, num_classes)

    def strip_classifier(self):
        return nn.Sequential(self.base_nn, self.adapt_nn)

    def invariant_features(self, x):
        return self.adapt_nn(self.base_nn(x))

    def forward(self, x):
        return self.classifier(self.invariant_features(x))


def train_width(width, src_loader, tgt_loader, log_dir, num_epochs=10, lr=0.0001):
    log_path = os.path.join(log_dir, f'width{width}.csv')
    logger = TrainingLogger(log_path, ['epoch', 'source_acc', 'test_acc'])
    device = get_device()

    model = DomainConfusion(pretrained_alexnet_fc7(), base_width=4098, adapt_width=width, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        _, src_acc = train_epoch(model, )


if __name__ == '__main__':
    # set up the source and target datasets
    root_dir = '/home/mchobanyan/data/research/transfer/office'

    amazon_data = Office31(root_dir, domain='amazon', source=True, transforms=DefaultTransform())
    webcam_data = Office31(root_dir, domain='webcam', source=False, transforms=DefaultTransform())

    amazon_loader = DataLoader(amazon_data, batch_size=BATCH_SIZE, shuffle=True)
    webcam_loader = DataLoader(webcam_data, batch_size=BATCH_SIZE, shuffle=True)

    widths = [2 ** i for i in range(6, 13)]
    print(DomainConfusion(pretrained_alexnet_fc7(), 4098, 256, 31))
