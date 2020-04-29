# Deep Domain Confusion

### Paper:
[Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf)

### Description:
A domain adaptation approach where a new, bottleneck adaptation layer is introduced into a pre-trained CNN, which then feeds
into an auxiliary domain confusion loss (based on maximum mean discrepancy). Training the network to lower both the standard label cross entropy loss
and the domain confusion loss enables us to create features which are both invariant to the domain and discriminative towards the labels.

### Things that have been changed:
- The optimizer was not specified in the paper (here we use SGD with base lr=1e-5, momentum=0.9)
- For the adaption layer width search, each model is trained for 30 epochs.

### To Do:
- Implement semi-supervised domain adaptation as well
- Add "Frustratingly easy domain adaptation" to the adaptation layer position search