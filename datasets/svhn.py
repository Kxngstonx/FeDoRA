import os
import numpy as np
from torchvision.datasets import SVHN


class SVHN_truncated(SVHN):
    """Thin wrapper around torchvision.datasets.SVHN that exposes the
    ``targets`` / ``num_classes`` interface expected by ``partition_data``
    in utils.py.
    """
    num_classes = 10

    def __init__(self, root, train=True, transform=None, download=True):
        split = 'train' if train else 'test'
        super().__init__(root=root, split=split, transform=transform, download=download)
        # torchvision SVHN stores labels in self.labels (numpy array)
        self.targets = self.labels.tolist()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target
