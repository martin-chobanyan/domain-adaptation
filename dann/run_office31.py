# TODO: figure out the number of epochs to train

import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from domain_adapt.data import ImagenetNorm
from domain_adapt.data.office31 import Office31
from domain_adapt.utils.misc import create_dir, get_script_dir
from domain_adapt.utils.train import TrainingLogger, test_epoch, checkpoint

from model import *

NUM_CLASSES = 31
BATCH_SIZE = NUM_CLASSES  # this is just a convenience so that the dataset is divisible by the batch size
NUM_RUNS = 1
NUM_EPOCHS = 100
MOMENTUM = 0.9
IMAGE_SHAPE = (256, 256)


def load_model():
    # set up the model components as specified in the paper
    feature_nn = pretrained_alexnet()
    label_nn = LabelClassifier(NUM_CLASSES)
    domain_nn = DomainClassifier()
    return DANN(feature_nn, label_nn, domain_nn)


def run_domain_adaptation(root_dir,
                          output_dir,
                          source_domain,
                          target_domain,
                          device,
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

    # set up the learning rate and domain adaptation hyperparameter schedulers
    schedule_lr = LRScheduler(num_epochs)
    schedule_da = DAScheduler(num_epochs)

    criterion = CrossEntropyLoss()
    for run in range(num_runs):
        src_data = Office31(root_dir, domain=source_domain, source=True, transforms=transforms)
        tgt_data = Office31(root_dir, domain=target_domain, source=False, transforms=transforms)

        src_loader = DataLoader(src_data, source_batch_size, shuffle=True)
        tgt_loader = DataLoader(tgt_data, target_batch_size, shuffle=True)

        # set up the DANN model
        model = load_model()
        model = model.to(device)

        for epoch in tqdm(range(num_epochs)):
            # set the optimizer's new learning rate (this wrapper operation is cheap)
            learning_rate = schedule_lr(epoch)
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)

            # train the model
            da_lambda = schedule_da(epoch)
            # da_lambda = epoch / num_epochs

            # print(f'Epoch: {epoch}')
            # print(f'Learning Rate: {learning_rate}')
            # print(f'DA lambda: {da_lambda}')
            # print()

            train_dann_epoch(model, src_loader, tgt_loader, criterion, optimizer, device, da_lambda)

            # test the label classifier
            full_classifier = model.retrieve_model()
            src_loss, src_acc = test_epoch(full_classifier, src_loader, criterion, device)
            tgt_loss, tgt_acc = test_epoch(full_classifier, tgt_loader, criterion, device)

            logger.add_entry(run, epoch, src_loss, tgt_loss, src_acc, tgt_acc)

        if save_models:
            checkpoint(model, os.path.join(model_dir, f'model-run{run}.pt'))


if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the directory path where this file exists
    output_dir = os.path.join(get_script_dir(__file__), 'results', 'office-31')
    create_dir(output_dir)

    img_transforms = Compose([Resize(IMAGE_SHAPE), ToTensor(), ImagenetNorm()])

    print('amazon -> webcam:')
    run_domain_adaptation(root_dir, output_dir, 'amazon', 'webcam', device, transforms=img_transforms)
    print()
