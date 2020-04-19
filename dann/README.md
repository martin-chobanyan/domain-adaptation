# Domain Adversarial Neural Network (DANN)

### Paper:
[Domain-Adversarial Training of Neural Networks](https://arxiv.org/pdf/1505.07818.pdf)

### Description:
A domain adaptation approach where a neural network is adversarially trained to detect features 
which are discriminative towards the task and invariant towards the domain.

### To Do:
- add script for MNIST-M adaptation
- add script for Amazon Reviews adaptation

### Things that have been changed:
- number of epochs run is set to 100
- image processing is normalized by channel mean and std dev
- data loading follows standard Office-31 training style