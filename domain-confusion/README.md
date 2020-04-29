# Deep Domain Confusion

### Paper:
[Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf)

### Description:
A domain adaptation approach where a new adaptation layer is introduced into a pre-trained CNN, which then feeds
into an auxiliary domain confusion loss (based on maximum mean discrepancy). Training the network to lower both the standard label cross entropy loss
and the domain confusion loss forces the source and target distributions to have closer features spaces 
(while not hindering their ability to discriminate with respect to the labels)

### Things that have been changed:
- The optimizer and learning rate were not specified in the paper (here we use SGD with base lr=1e-5, momentum=0.9)
- For the adaption layer width search, each model is trained for 30 epochs.

### To Do:
- Implement semi-supervised domain adaptation as well