"""This script searches for the best width of the adaptation layer attached to fc7 in AlexNet """
import os

import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adapt.data import Office31
from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.nn.models import pretrained_alexnet_fc7
from domain_adapt.utils.misc import create_dir, get_device, get_script_dir
from domain_adapt.utils.train import TrainingLogger, train_epoch, test_epoch

from model import calculate_mmd, load_model

BATCH_SIZE = 64
NUM_CLASSES = 31
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
FIGSIZE = (9, 6)


def train_width(width, src_loader, tgt_loader, log_dir, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    # the accuracies are logged for a sanity check
    log_path = os.path.join(log_dir, f'width{width}.csv')
    logger = TrainingLogger(log_path, ['epoch', 'src_acc', 'tgt_acc'])

    device = get_device()
    model = load_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        _, src_acc = train_epoch(model, src_loader, criterion, optimizer, device)
        _, tgt_acc = test_epoch(model, tgt_loader, criterion, device)
        logger.add_entry(epoch, src_acc, tgt_acc)

    mmd_val = calculate_mmd(model.strip_classifier(), src_loader, tgt_loader, device, progress=False)
    return mmd_val


def plot_results(widths, mmds, accs, figsize=FIGSIZE):
    idx = list(range(len(widths)))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(idx, accs, color='red', label='Test accuracy')
    ax1.set_xticks(idx)
    ax1.set_xticklabels(widths)
    ax1.set_xlabel('Adaption layer widths')
    ax1.set_ylabel('Accuracy', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(idx, mmds, '--', color='blue', label='Maximum mean discrepancy')
    ax2.set_ylabel('Maximum Mean Discrepancy', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # set up the source and target datasets
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    amazon_data = Office31(root_dir, domain='amazon', transforms=DefaultTransform())
    webcam_data = Office31(root_dir, domain='webcam', transforms=DefaultTransform())
    amazon_loader = DataLoader(amazon_data, batch_size=BATCH_SIZE, shuffle=True)
    webcam_loader = DataLoader(webcam_data, batch_size=BATCH_SIZE, shuffle=True)

    # set up the output directory
    out_dir = os.path.join(get_script_dir(__file__), 'results', 'width-search')
    create_dir(out_dir)

    # search the widths and save their source-target MMD values
    mmd_vals = []
    widths = [2 ** i for i in range(6, 13)]
    for w in tqdm(widths):
        mmd_vals.append(train_width(w, amazon_loader, webcam_loader, out_dir))
    mmd_df = DataFrame({'width': widths, 'mmd': mmd_vals})
    mmd_df.to_csv(os.path.join(out_dir, 'mmd-values.csv'), index=False)

    # retrieve the target accuracies for the last epoch
    acc_vals = []
    for w in widths:
        acc_df = read_csv(os.path.join(out_dir, f'width{w}.csv'))
        acc_vals.append(acc_df.iloc[-1]['tgt_acc'])

    plot_results(widths, mmd_vals, acc_vals)
