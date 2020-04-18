from torchvision.transforms import Normalize

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDEVS = [0.229, 0.224, 0.225]


class ImagenetNorm(Normalize):
    def __init__(self):
        super().__init__(mean=IMAGENET_MEANS, std=IMAGENET_STDEVS)
