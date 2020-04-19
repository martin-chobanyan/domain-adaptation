from torch.nn import Sequential
from torchvision.models import alexnet


# ----------------------------------------------------------------------------------------------------------------------
# Pre-trained feature extractors
# ----------------------------------------------------------------------------------------------------------------------

def _pretrained_alexnet(layer='fc8'):
    model = alexnet(pretrained=True)
    if layer == 'fc8':
        return model
    elif layer == 'fc6':
        layer_idx = 3
    elif layer == 'fc7':
        layer_idx = 6
    else:
        raise ValueError(f'Layer {layer} is not supported for alexnet.')
    model.classifier = Sequential(*[child for child in list(model.classifier.children())[:layer_idx]])
    return model


def pretrained_alexnet_fc6():
    """Prepares an AlexNet feature extractor ending at layer fc6 (pre-trained on ImageNet)

    Returns
    -------
    nn.Module
    """
    return _pretrained_alexnet('fc6')


def pretrained_alexnet_fc7():
    """Prepares an AlexNet feature extractor ending at layer fc7 (pre-trained on ImageNet)

    Returns
    -------
    nn.Module
    """
    return _pretrained_alexnet('fc7')


def pretrained_alexnet_fc8():
    """Prepares an AlexNet feature extractor with every layer (pre-trained on ImageNet)

    Returns
    -------
    nn.Module
    """
    return _pretrained_alexnet('fc8')
