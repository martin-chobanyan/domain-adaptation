"""This script searches for the most appropriate layer (based on MMD) in AlexNet to attach the adaptation layer"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adapt.data.office31 import Office31
from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.nn.loss import mmd
from domain_adapt.nn.models import pretrained_alexnet_fc6, pretrained_alexnet_fc7, pretrained_alexnet_fc8
from domain_adapt.utils.misc import get_device, load_batch

BATCH_SIZE = 32


def calculate_mmd(model, src_loader, tgt_loader, device):
    model = model.eval()
    model = model.to(device)

    mmd_values = []
    for src_imgs, _ in tqdm(src_loader):
        tgt_imgs, _ = load_batch(tgt_loader)

        src_imgs = src_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)

        src_features = model(src_imgs)
        tgt_features = model(tgt_imgs)

        val = mmd(src_features, tgt_features)
        mmd_values.append(val)
    return sum(mmd_values) / len(mmd_values)


def search_fc_layers(src_loader, tgt_loader):
    device = get_device()

    print('Calculating MMD for fc6:')
    model = pretrained_alexnet_fc6()
    mmd_fc6 = calculate_mmd(model, src_loader, tgt_loader, device)
    print()

    print('Calculating MMD for fc7:')
    model = pretrained_alexnet_fc7()
    mmd_fc7 = calculate_mmd(model, src_loader, tgt_loader, device)
    print()

    print('Calculating MMD for fc8:')
    model = pretrained_alexnet_fc8()
    mmd_fc8 = calculate_mmd(model, src_loader, tgt_loader, device)
    print()

    return [mmd_fc6, mmd_fc7, mmd_fc8]


if __name__ == '__main__':
    # set up the source and target datasets
    root_dir = '/home/mchobanyan/data/research/transfer/office'

    amazon_data = Office31(root_dir, domain='amazon', source=True, transforms=DefaultTransform())
    webcam_data = Office31(root_dir, domain='webcam', source=False, transforms=DefaultTransform())

    amazon_loader = DataLoader(amazon_data, batch_size=BATCH_SIZE, shuffle=True)
    webcam_loader = DataLoader(webcam_data, batch_size=BATCH_SIZE, shuffle=True)

    result = search_fc_layers(amazon_loader, webcam_loader)
    for layer, val in zip(['fc6', 'fc7', 'fc8'], result):
        print(f'{layer}: {val}')
