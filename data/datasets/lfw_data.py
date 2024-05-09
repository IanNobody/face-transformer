import random

import PIL.Image as Image
from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
import torch
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    def __init__(self, transform=None, device=None):
        lfw_pairs = fetch_lfw_pairs(color=True, funneled=True)

        self.pairs = lfw_pairs.pairs
        self.labels = lfw_pairs.target

        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = (self.pairs[idx][0] * 255).astype(np.uint8)
        img2 = (self.pairs[idx][1] * 255).astype(np.uint8)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {"img1": img1, "img2": img2, "label": self.labels[idx]}

    def num_of_classes(self):
        return 5749

    @staticmethod
    def collate_fn(batch):
        img1 = torch.stack([item["img1"] for item in batch])
        img2 = torch.stack([item["img2"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])

        return {"img1": img1, "img2": img2, "label": labels}
