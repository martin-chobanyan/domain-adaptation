# Deep Domain Confusion

### Paper:
[Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf)

### Description:
A domain adaptation approach where a new adaptation layer is introduced into a pre-trained CNN, which then feeds
into an auxiliary domain confusion loss (based on maximum mean discrepancy). Training the network to lower both the standard label cross entropy loss
and the domain confusion loss forces the source and target distributions to have closer features spaces 
(while not hindering their ability to discriminate with respect to the labels)

### To Do:
- add search for target layer
- add search for adaptation layer dimension

### Things that have been changed:
- N/A