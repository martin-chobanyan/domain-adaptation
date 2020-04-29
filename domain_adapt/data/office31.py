import os
import random

from torchvision.datasets import ImageFolder

VALID_DOMAINS = ['amazon', 'dslr', 'webcam']


class Office31(ImageFolder):
    def __init__(self, root_dir, domain, full=True, num_per_category=None, transforms=None):
        # store the attributes
        self.root_dir = root_dir
        self.domain = self.__check_domain(domain)
        self.domain_dir = os.path.join(self.root_dir, self.domain, 'images')
        self.full = full
        self.num_per_category = num_per_category
        self.transforms = transforms

        # set up either the full train protocol or sampling protocol
        if self.full and self.num_per_category is None:
            super().__init__(root=self.domain_dir, transform=transforms)
        else:
            if self.num_per_category is None:
                self.num_per_category = self.__default_num_per_category(domain)
            self.sampled_files = self.sample_per_category()
            super().__init__(root=self.domain_dir, is_valid_file=self.check_file, transform=transforms)

    @staticmethod
    def __check_domain(domain):
        domain = domain.lower()
        if domain not in VALID_DOMAINS:
            raise ValueError(f'"domain" argument must be one of: {VALID_DOMAINS}')
        return domain

    @staticmethod
    def __default_num_per_category(domain):
        if domain == 'amazon':
            n = 20
        else:
            n = 8
        return n

    def sample_per_category(self):
        sampled_files = dict()
        for category in os.listdir(self.domain_dir):
            filenames = os.listdir(os.path.join(self.domain_dir, category))
            num_files = len(filenames)
            if self.num_per_category < num_files:
                sample_result = random.sample(filenames, self.num_per_category)
            else:
                sample_result = filenames
            sampled_files[category] = set(sample_result)
        return sampled_files

    def check_file(self, filepath):
        # isolate the name of the file and its image category
        category_path, filename = os.path.split(filepath)
        category = os.path.basename(category_path)

        # check if the input file is one of the sampled files for its category
        file_sample = self.sampled_files[category]
        return (filename in file_sample)
