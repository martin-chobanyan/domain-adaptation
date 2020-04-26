# domain-adaptation

This repo provides pytorch implementations of several domain adaptation techniques,
along with scripts recreating the results of their respective papers.

It also comes with a local python package **domain_adapt** which holds several utilities
for domain adaptation including: 
- Loaders for common domain adaptation benchmark datasets
- Common layers and modeling tools for domain adaptation

## Installation
Run `pip install .` from the repo's root to install the **domain_adapt** package.

## Dependencies
- matplotlib
- numpy
- pandas
- torch
- torchvision
- tqdm

## Papers to implement
- [Domain Adversarial Neural Networks (DANN)](https://arxiv.org/pdf/1505.07818.pdf)
- [Deep Domain Confusion](https://arxiv.org/pdf/1412.3474.pdf)
- [Deep CORAL](https://arxiv.org/abs/1607.01719)

## Datasets to add
- Office-10
- VisDA
- Amazon Review