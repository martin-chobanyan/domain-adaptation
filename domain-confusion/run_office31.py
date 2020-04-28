import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adapt.data import Office31
from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.utils.misc import create_dir, get_device, get_script_dir
from domain_adapt.utils.train import checkpoint, test_epoch, TrainingLogger

from model import DomainConfusion, load_model, train_domain_confusion

NUM_RUNS = 5
NUM_EPOCHS = 50
BATCH_SIZE = 32
ADAPT_WIDTH = 256
# DA_LAMBDA = 0.25
DA_LAMBDA = 0
LEARNING_RATE = 0.0001


def run_domain_adaptation(root_dir,
                          output_dir,
                          source_domain,
                          target_domain,
                          adapt_width=ADAPT_WIDTH,
                          num_runs=NUM_RUNS,
                          num_epochs=NUM_EPOCHS,
                          source_batch_size=BATCH_SIZE,
                          target_batch_size=BATCH_SIZE,
                          save_models=False,
                          transforms=None):
    # set up the train logger
    logger_path = os.path.join(output_dir, f'{source_domain}-{target_domain}.csv')
    logger = TrainingLogger(logger_path)

    # set up the model checkpoint directory
    model_dir = os.path.join(output_dir, 'models', f'{source_domain}-{target_domain}')
    if save_models:
        create_dir(model_dir)

    criterion = nn.CrossEntropyLoss()
    for run in tqdm(range(num_runs)):
        src_data = Office31(root_dir, domain=source_domain, transforms=transforms, full=False)
        tgt_data = Office31(root_dir, domain=target_domain, transforms=transforms, full=True)

        src_loader = DataLoader(src_data, source_batch_size, shuffle=True)
        tgt_loader = DataLoader(tgt_data, target_batch_size, shuffle=True)

        # set up the DANN model
        device = get_device()
        model = load_model(adapt_width)
        model = model.to(device)
        optimizer = Adam([{'params': model.base_nn.parameters()},
                          {'params': model.adapt_nn.parameters(), 'lr': 10 * LEARNING_RATE},
                          {'params': model.classifier.parameters(), 'lr': 10 * LEARNING_RATE}],
                         lr=LEARNING_RATE)

        for epoch in range(num_epochs):
            train_domain_confusion(model, src_loader, tgt_loader, criterion, optimizer, device, da_lambda=DA_LAMBDA)
            src_loss, src_acc = test_epoch(model, src_loader, criterion, device)
            tgt_loss, tgt_acc = test_epoch(model, tgt_loader, criterion, device)
            logger.add_entry(run, epoch, src_loss, tgt_loss, src_acc, tgt_acc)

        if save_models:
            checkpoint(model, os.path.join(model_dir, f'model-run{run}.pt'))


if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'

    # get the directory path where this file exists
    output_dir = os.path.join(get_script_dir(__file__), 'results-base', 'office-31')
    create_dir(output_dir)

    print('amazon -> webcam:')
    run_domain_adaptation(root_dir, output_dir, 'amazon', 'webcam', transforms=DefaultTransform())
    print()

    # print('dslr -> webcam:')
    # run_domain_adaptation(root_dir, output_dir, 'dslr', 'webcam', transforms=DefaultTransform())
    # print()
    #
    # print('webcam -> dslr:')
    # run_domain_adaptation(root_dir, output_dir, 'webcam', 'dslr', transforms=DefaultTransform())
    # print('\nDone!')
