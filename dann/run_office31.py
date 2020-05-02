import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adapt.data.transforms import DefaultTransform
from domain_adapt.data.office31 import Office31
from domain_adapt.utils.misc import create_dir, get_script_dir
from domain_adapt.utils.train import TrainingLogger, test_epoch, checkpoint

from dann_model import load_model, train_dann_epoch, LRScheduler, DAScheduler

NUM_CLASSES = 31
BATCH_SIZE = 256
NUM_RUNS = 1
NUM_EPOCHS = 200
# LEARNING_RATE = 0.0001
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def run_domain_adaptation(root_dir,
                          output_dir,
                          src_domain,
                          tgt_domain,
                          device,
                          num_runs=NUM_RUNS,
                          num_epochs=NUM_EPOCHS,
                          src_batch_size=BATCH_SIZE,
                          tgt_batch_size=BATCH_SIZE,
                          save_models=False,
                          transforms=None):
    # set up the train logger
    logger_path = os.path.join(output_dir, f'{src_domain}-{tgt_domain}.csv')
    header = ['run', 'epoch', 'src_loss', 'domain_loss', 'tgt_loss', 'src_acc', 'tgt_acc']
    logger = TrainingLogger(logger_path, header=header)

    # set up the learning rate and domain adaptation hyperparameter schedulers
    schedule_lr = LRScheduler(num_epochs, init_lr=LEARNING_RATE)
    schedule_da = DAScheduler(num_epochs)

    criterion = CrossEntropyLoss()
    for run in range(num_runs):
        src_data = Office31(root_dir, domain=src_domain, transforms=transforms)
        tgt_data = Office31(root_dir, domain=tgt_domain, transforms=transforms)

        src_loader = DataLoader(src_data, src_batch_size, shuffle=True)
        tgt_loader = DataLoader(tgt_data, tgt_batch_size, shuffle=True)

        # set up the DANN model
        model = load_model(NUM_CLASSES)
        model = model.to(device)

        for epoch in tqdm(range(num_epochs), desc=f'{src_domain}->{tgt_domain}, Run {run}'):
            # set the optimizer's new learning rate (this wrapper operation is cheap)
            learning_rate = schedule_lr(epoch)
            # learning_rate = LEARNING_RATE
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)

            # train the model
            da_lambda = schedule_da(epoch)
            src_acc, src_loss, domain_loss = train_dann_epoch(model, src_loader, tgt_loader,
                                                    criterion, optimizer, device, da_lambda)

            # test the label classifier
            full_classifier = model.get_label_classifier()
            tgt_loss, tgt_acc = test_epoch(full_classifier, tgt_loader, criterion, device)

            logger.add_entry(run, epoch, src_loss, domain_loss, tgt_loss, src_acc, tgt_acc)

        if save_models:
            model_dir = os.path.join(output_dir, 'models', f'{src_domain}-{tgt_domain}')
            create_dir(model_dir)
            checkpoint(model, os.path.join(model_dir, f'model-run{run}.pt'))


if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the directory path where this file exists
    output_dir = os.path.join(get_script_dir(__file__), 'results4', 'office-31')
    create_dir(output_dir)

    run_domain_adaptation(root_dir, output_dir, 'amazon', 'webcam', device, transforms=DefaultTransform())
