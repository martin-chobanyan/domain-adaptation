# For each of 20 random splits
# if the domain is amazon, there will be 20 examples per category
# if the domain is dslr/webcam, there will be 8 examples per category
# the target will always have 3 examples per category

import os
import random

from torchvision.datasets import ImageFolder

VALID_DOMAINS = ['amazon', 'dslr', 'webcam']


class CategorySampler:
    def __init__(self, data_dir, n_per_category):
        self.data_dir = data_dir
        self.n_per_category = n_per_category
        self.sampled_files = self.collect_sample()

    def collect_sample(self):
        sampled_files = dict()
        for category in os.listdir(self.data_dir):
            category_dir = os.path.join(self.data_dir, category)
            sampled_files[category] = set(random.sample(os.listdir(category_dir), self.n_per_category))
        return sampled_files

    def __call__(self, filepath):
        # isolate the name of the file and its image category
        category_path, filename = os.path.split(filepath)
        category = os.path.basename(category_path)

        # check if the input file is one of the sampled files for its category
        file_sample = self.sampled_files[category]
        return (filename in file_sample)


class Office31(ImageFolder):
    def __init__(self, root_dir, domain, source, per_category_config=None):
        if domain.lower() not in VALID_DOMAINS:
            raise ValueError(f'"domain" argument must be one of: {VALID_DOMAINS}')

        self.root_dir = root_dir
        self.domain = domain
        self.source = source
        self.domain_dir = os.path.join(root_dir, domain, 'images')

        super().__init__(root=self.domain_dir, is_valid_file=CategorySampler(self.domain_dir, 2))
