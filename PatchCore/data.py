import os
from os.path import isdir
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np

from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


DATASETS_PATH = Path("./datasets")
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])


class LoadDataset:
    def __init__(self, cls : str, size : int):
        self.cls = cls
        self.size = size
        print('size:', size)
        self.train_ds = TrainDataset(cls, size)
        self.test_ds = TestDataset(cls, size)

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


class TrainDataset(ImageFolder):
    def __init__(self, cls : str, size : int):
        super().__init__(
            root=DATASETS_PATH / cls / "train",
            transform=transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                # transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )

        self.cls = cls
        self.size = size

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:
            sample_class = 0
        else:
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_class


class TestDataset(ImageFolder):
    def __init__(self, cls : str, size : int):
        super().__init__(
            root=DATASETS_PATH / cls / "test",
            transform=transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                # transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
        )
        self.cls = cls
        self.size = size

            
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        
        if "good" in path:
            sample_class = 0
        else:
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_class
