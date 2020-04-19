from torch.nn import Sequential
from torchvision.models import alexnet


def pretrained_alexnet_fc7():
    """Prepares an AlexNet feature extractor ending at layer fc7 (pre-trained on ImageNet)

    Returns
    -------
    nn.Module
    """
    model = alexnet(pretrained=True)
    model.classifier = Sequential(*[child for child in list(model.classifier.children())[:6]])
    return model
