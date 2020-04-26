import torch.nn as nn
from tqdm import tqdm

from domain_adapt.nn.loss import mmd
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
