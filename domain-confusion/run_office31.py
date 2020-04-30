"""Training script for deep domain confusion on Office-31 dataset"""
import os

import pandas as pd
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adapt.data import Office31
from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.utils.misc import create_dir, get_device, get_script_dir
from domain_adapt.utils.train import checkpoint, test_epoch, TrainingLogger

from dc_model import load_model, train_domain_confusion

NUM_RUNS = 5
NUM_EPOCHS = 150
BATCH_SIZE = 256
ADAPT_WIDTH = 256
DA_LAMBDA = 0.25
LEARNING_RATE = 0.00001
MOMENTUM = 0.9


def run_domain_adaptation(root_dir,
                          output_dir,
                          src_domain,
                          tgt_domain,
                          adapt_width=ADAPT_WIDTH,
                          num_runs=NUM_RUNS,
                          num_epochs=NUM_EPOCHS,
                          src_batch_size=BATCH_SIZE,
                          tgt_batch_size=BATCH_SIZE,
                          save_models=False,
                          transforms=None):
    # set up the train logger
    logger_path = os.path.join(output_dir, f'{src_domain}-{tgt_domain}.csv')
    logger = TrainingLogger(logger_path)

    criterion = nn.CrossEntropyLoss()
    for run in range(num_runs):
        src_data = Office31(root_dir, domain=src_domain, transforms=transforms, full=False)
        tgt_data = Office31(root_dir, domain=tgt_domain, transforms=transforms, full=True)

        src_loader = DataLoader(src_data, src_batch_size, shuffle=True)
        tgt_loader = DataLoader(tgt_data, tgt_batch_size, shuffle=True)

        # set up the model
        device = get_device()
        model = load_model(adapt_width)
        model = model.to(device)
        optimizer = SGD([{'params': model.base_nn.parameters()},
                         {'params': model.adapt_nn.parameters(), 'lr': 10 * LEARNING_RATE},
                         {'params': model.classify_nn.parameters(), 'lr': 10 * LEARNING_RATE}],
                        lr=LEARNING_RATE, momentum=MOMENTUM)

        # train and test the model
        for epoch in tqdm(range(num_epochs), desc=f'{src_domain}->{tgt_domain}, Run {run}'):
            src_loss, src_acc = train_domain_confusion(model, src_loader, tgt_loader, criterion, optimizer, device,
                                                       da_lambda=DA_LAMBDA)
            tgt_loss, tgt_acc = test_epoch(model, tgt_loader, criterion, device)
            logger.add_entry(run, epoch, src_loss, tgt_loss, src_acc, tgt_acc)

        # optionally save the model parameters
        if save_models:
            model_dir = os.path.join(output_dir, 'models', f'{src_domain}-{tgt_domain}')
            create_dir(model_dir)
            checkpoint(model, os.path.join(model_dir, f'model-run{run}.pt'))


if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'

    # get the directory path where this file exists
    output_dir = os.path.join(get_script_dir(__file__), 'results', 'office-31')
    create_dir(output_dir)

    print('Running Deep Domain Confusion on Office-31...')
    run_domain_adaptation(root_dir, output_dir, 'amazon', 'webcam', transforms=DefaultTransform())
    run_domain_adaptation(root_dir, output_dir, 'dslr', 'webcam', transforms=DefaultTransform())
    run_domain_adaptation(root_dir, output_dir, 'webcam', 'dslr', transforms=DefaultTransform())
    print('Done!')
    print(f'Training logs can be found in {output_dir}')
    print()

    print(f'Average target accuracies across all {NUM_RUNS} runs')
    for filename in os.listdir(output_dir):
        # get the source and target domains
        base, _ = os.path.splitext(filename)
        src_domain, tgt_domain = base.split('-')
        src_domain = src_domain[0].upper()
        tgt_domain = tgt_domain[0].upper()

        # average the target accuracy across all runs
        log_df = pd.read_csv(os.path.join(output_dir, filename))
        avg_acc = log_df.loc[log_df['epoch'] == NUM_EPOCHS - 1, 'tgt_acc'].mean()
        print(f'{src_domain}->{tgt_domain}:\t{round(100 * avg_acc, 4)}%')
