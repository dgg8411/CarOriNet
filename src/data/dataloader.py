import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

import configs.config as config


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    train_set = datasets.ImageFolder(os.path.join(config.DATASET_PATH, "train"), transform=transform)
    val_set   = datasets.ImageFolder(os.path.join(config.DATASET_PATH, "val"), transform=transform)
    test_set  = datasets.ImageFolder(os.path.join(config.DATASET_PATH, "test"), transform=transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_set.classes