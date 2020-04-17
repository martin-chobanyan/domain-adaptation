import os
import random

from torchvision.datasets import ImageFolder

VALID_DOMAINS = ['amazon', 'dslr', 'webcam']


class Office31(ImageFolder):
    def __init__(self, root_dir, domain, source, num_per_category=None):
        self.domain = domain.lower()
        if self.domain not in VALID_DOMAINS:
            raise ValueError(f'"domain" argument must be one of: {VALID_DOMAINS}')

        self.root_dir = root_dir
        self.source = source
        self.domain_dir = os.path.join(self.root_dir, self.domain, 'images')

        self.num_per_category = num_per_category
        if self.num_per_category is None:
            if source:
                if domain == 'amazon':
                    self.num_per_category = 20
                else:
                    self.num_per_category = 8
            else:
                self.num_per_category = 3

        self.sampled_files = self.sample_per_category()

        super().__init__(root=self.domain_dir, is_valid_file=self.check_file)

    def sample_per_category(self):
        sampled_files = dict()
        for category in os.listdir(self.domain_dir):
            category_dir = os.path.join(self.domain_dir, category)
            sampled_files[category] = set(random.sample(os.listdir(category_dir), self.num_per_category))
        return sampled_files

    def check_file(self, filepath):
        # isolate the name of the file and its image category
        category_path, filename = os.path.split(filepath)
        category = os.path.basename(category_path)

        # check if the input file is one of the sampled files for its category
        file_sample = self.sampled_files[category]
        return (filename in file_sample)
