import numpy
from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
import numpy as np
from os.path import join
from os import listdir
from os.path import isfile
from os.path import isdir
import tarfile
from zipfile import ZipFile
from PIL import Image


class VGGFaceDataset(Dataset):
    def __init__(self, img_dir, file_list_path, transform=None, augmentation=None, device=None):
        self.img_root = img_dir

        with open(file_list_path) as f:
            self.img = [line for line in f.read().splitlines() if isfile(join(img_dir, line))]

        self.ann = [int(img.split('/')[-2].removeprefix("n")) for img in self.img]
        self.transform = transform
        self.augmentation = augmentation
        self.device = device

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = join(self.img_root, self.img[idx])
        image = cv2.imread(img_path) / 255.0
        ann = torch.tensor(self.ann[idx])

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        if self.device:
            image = image.to(self.device)
            ann = ann.to(self.device)

        sample = {'image': image, 'entity': ann}
        return sample

    def num_of_classes(self):
        return max(self.ann) + 1
