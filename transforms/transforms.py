import numpy as np
import albumentations.augmentations.transforms as albu
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose

from albumentations.core.transforms_interface import ImageOnlyTransform


def train_transform(resize, normalize=None):
    if normalize == 'imagenet':
        trans_fucn = [
            albu.VerticalFlip(p=0.5),
            albu.HorizontalFlip(p=0.5),
            # albu.ToFloat(max_value=255, p=1.0),
            albu.Normalize(p=1.0),
            ToTensorV2(p=1.0)
        ]
    elif normalize == 'global_norm':
        trans_fucn = [
            albu.VerticalFlip(p=0.5),
            albu.HorizontalFlip(p=0.5),
            GlobalNormalize(p=1.0),
            # albu.ToFloat(max_value=255, p=1.0),
            ToTensorV2(p=1.0)
        ]
    else:
        trans_fucn = [
            albu.VerticalFlip(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            # albu.ToFloat(max_value=255, p=1.0),
            ToTensorV2(p=1.0)
        ]
    return Compose(trans_fucn, p=1.0)


def eval_transform(resize, normalize=None):
    if normalize == 'imagenet':
        trans_func = [
            albu.Normalize(p=1.0),
            ToTensorV2(p=1.0)
        ]
    elif normalize == 'global_norm':
        trans_func = [
            GlobalNormalize(p=1.0),
            ToTensorV2(p=1.0)
        ]
    else:
        trans_func = [
            albu.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(p=1.0)
        ]
    return Compose(trans_func, p=1.0)


def global_normalize(img):
    g_mean = np.mean(img, axis=(0, 1), dtype=np.float32)
    g_std = np.std(img, axis=(0, 1), dtype=np.float32)
    g_std[g_std==0] = 1.

    denominator = np.reciprocal(g_std, dtype=np.float32)

    img -= g_mean
    img *= denominator

    return img


class GlobalNormalize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(GlobalNormalize, self).__init__(always_apply, p)
    def apply(self, image, **params):
        return  global_normalize(image)
    def get_transform_init_args_names(self):
        return ()