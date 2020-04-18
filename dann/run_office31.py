# TODO: add training pipeline for DANN model
# TODO: figure out the number of epochs to train

import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adapt.data.office31 import Office31
from domain_adapt.utils.misc import create_dir, get_script_dir
from domain_adapt.utils.train import TrainingLogger
from model import pretrained_alexnet, LabelClassifier, DomainClassifier, DANN

NUM_CLASSES = 31
BATCH_SIZE = NUM_CLASSES  # this is just a convenience so that the dataset is divisible by the batch size
NUM_RUNS = 20
NUM_EPOCHS = 100


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
                          save_models=False):
    # set up the DANN model
    model = load_model()
    model = model.to(device)

    # set up the loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # set up the train logger
    logger_path = os.path.join(output_dir, f'{source_domain}-{target_domain}.csv')
    logger = TrainingLogger(logger_path)

    # set up the model checkpoint directory
    model_dir = os.path.join(output_dir, 'models')
    if save_models:
        create_dir(model_dir)

    for run in tqdm(range(num_runs)):
        source_data = Office31(root_dir, domain=source_domain, source=True)
        target_data = Office31(root_dir, domain=target_domain, source=False)

        source_loader = DataLoader(source_data, source_batch_size, shuffle=True)
        target_loader = DataLoader(target_data, target_batch_size, shuffle=True)

        for epoch in range(num_epochs):
            pass


if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the directory path where this file exists
    output_dir = os.path.join(get_script_dir(__file__), 'results', 'office-31')
    create_dir(output_dir)

    # amazon -> webcam
    run_domain_adaptation(root_dir, output_dir, source_domain='amazon', target_domain='webcam', device=device)
