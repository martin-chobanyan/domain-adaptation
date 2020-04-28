import torch.nn as nn
from tqdm import tqdm

from domain_adapt.nn.loss import mmd
from domain_adapt.nn.models import pretrained_alexnet_fc7
from domain_adapt.utils.misc import load_batch


class DomainConfusion(nn.Module):
    def __init__(self, base_nn, base_width, adapt_width, num_classes):
        super().__init__()
        self.base_nn = base_nn
        self.adapt_nn = nn.Linear(base_width, adapt_width)
        self.classifier = nn.Linear(adapt_width, num_classes)

    def strip_classifier(self):
        return nn.Sequential(self.base_nn, self.adapt_nn)

    def invariant_features(self, x):
        return self.adapt_nn(self.base_nn(x))

    def forward(self, x):
        return self.classifier(self.invariant_features(x))


def load_model(width):
    return DomainConfusion(pretrained_alexnet_fc7(), base_width=4096, adapt_width=width, num_classes=31)


def calculate_mmd(model, src_loader, tgt_loader, device, progress=True):
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
    model = model.train()
    for src_images, src_labels in src_loader:
        tgt_images, _ = load_batch(tgt_loader)

        src_images = src_images.to(device)
        src_labels = src_labels.to(device)
        tgt_images = tgt_images.to(device)

        optimizer.zero_grad()

        # source label loss
        label_scores = model(src_images)
        loss_label = criterion(label_scores, src_labels)

        # domain confusion loss
        src_features = model.invariant_features(src_images)
        tgt_features = model.invariant_features(tgt_images)
        loss_mmd = mmd(src_features, tgt_features)

        loss_total = loss_label + da_lambda * loss_mmd
        loss_total.backward()
        optimizer.step()
