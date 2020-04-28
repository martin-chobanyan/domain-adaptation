from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from . import IMAGENET_MEANS, IMAGENET_STDEVS


class ImagenetNorm(Normalize):
    def __init__(self):
        super().__init__(mean=IMAGENET_MEANS, std=IMAGENET_STDEVS)


class DefaultTransform:
    def __init__(self, img_shape=256):
        self.transform = Compose([Resize(img_shape), ToTensor(), ImagenetNorm()])

    def __call__(self, img):
        return self.transform(img)
