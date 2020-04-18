# TODO: add training pipeline for DANN model
# TODO: figure out the number of epochs to train

import os

import torch
from tqdm import tqdm

from domain_adapt.data.office31 import Office31
from domain_adapt.utils.misc import create_dir, get_script_dir
from domain_adapt.utils.train import TrainingLogger
from model import pretrained_alexnet, LabelClassifier, DomainClassifier, DANN

NUM_CLASSES = 31
NUM_RUNS = 20
BATCH_SIZE = NUM_CLASSES  # this is just a convenience so that the dataset is divisible by the batch size


def load_model():
    # set up the model componenets
    feature_extractor = pretrained_alexnet()
    label_classifier = LabelClassifier(NUM_CLASSES)
    domain_classifier = DomainClassifier()

    return DANN(feature_extractor, label_classifier, domain_classifier)


def run_domain_adaptation(root_dir,
                          output_dir,
                          source_domain,
                          target_domain,
                          device,
                          num_runs=NUM_RUNS,
                          source_batch_size=BATCH_SIZE,
                          target_batch_size=BATCH_SIZE,
                          save_models=False):
    logger_path = os.path.join(output_dir, f'{source_domain}-{target_domain}.csv')
    logger = TrainingLogger(logger_path)

    model_dir = os.path.join(output_dir, 'models')
    if save_models:
        create_dir(model_dir)

    for run in tqdm(range(num_runs)):
        source_data = Office31(root_dir, domain=source_domain, source=True)
        target_data = Office31(root_dir, domain=target_domain, source=False)


if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the directory path where this file exists
    output_dir = os.path.join(get_script_dir(__file__), 'results', 'office-31')
    create_dir(output_dir)

    # amazon -> webcam
    run_domain_adaptation(root_dir, output_dir, source_domain='amazon', target_domain='webcam', device=device)
