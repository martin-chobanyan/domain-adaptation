from torch.nn import Sequential
from torchvision.models import alexnet


def pretrained_alexnet():
    """Prepares an AlexNet feature extractor pre-trained on ImageNet

    Note: The final fc8 layer is removed from the network, resulting in a 4096-dim output.

    Returns
    -------
    nn.Module
    """
    model = alexnet(pretrained=True)
    model.classifier = Sequential(*[child for child in list(model.classifier.children())[:6]])
    return model
