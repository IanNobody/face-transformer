import os

import cv2
from torch.utils.data import Dataset
import numpy as np
import torch


class CACDDataset(Dataset):
    def __init__(self, path, transform=None, augmentation=None, device=None):
        self.files = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith(".jpg")]
        self.classes = np.unique(["".join(filename.split("_")[1:-1]) for filename in self.files]).tolist()
        self.transform = transform
        self.augmentation = augmentation
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)
        label = self.classes.index("".join(filename.split("_")[1:-1]))

        image = cv2.imread(path) / 255.0
        ann = torch.tensor(label)

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        sample = {'data': image, 'entity': ann}
        return sample

    def num_of_classes(self):
        return len(self.classes)