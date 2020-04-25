"""This script searches for the most appropriate layer (based on MMD) in AlexNet to attach the adaptation layer"""

import torch
from torch.utils.data import DataLoader

from domain_adapt.data.office31 import Office31
from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.nn.loss import mmd
from domain_adapt.nn.models import pretrained_alexnet_fc6, pretrained_alexnet_fc7, pretrained_alexnet_fc8

BATCH_SIZE = 31

if __name__ == '__main__':
    # set up the source and target datasets
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    amazon_data = Office31(root_dir, domain='amazon', source=True, transforms=DefaultTransform())
    webcam_data = Office31(root_dir, domain='webcam', source=False, transforms=DefaultTransform())

    amazon_loader = DataLoader(amazon_data, batch_size=BATCH_SIZE, shuffle=True)
    webcam_loader = DataLoader(webcam_data, batch_size=BATCH_SIZE, shuffle=True)

    # calculate MMD for fc6
    model = pretrained_alexnet_fc6()
    model = model.eval()
    model = model.to(device)

    mmd_values = []
    for src_imgs, tgt_imgs in zip(amazon_loader, webcam_loader):
        src_imgs = src_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)
        mmd_values.append(mmd(src_imgs, tgt_imgs))
    mmd_val = sum(mmd_values) / len(mmd_values)

