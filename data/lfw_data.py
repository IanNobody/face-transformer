from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
import torch
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    def __init__(self, transform=None, device=None):
        lfw_pairs = fetch_lfw_pairs(color=True, funneled=True)

        pairs = lfw_pairs.pairs
        labels = lfw_pairs.target

        same_face_pairs = pairs[labels == 1]
        diff_face_pairs = pairs[labels == 0]
        min_size = min(len(same_face_pairs), len(diff_face_pairs))
        same_face_pairs = same_face_pairs[:min_size]
        diff_face_pairs = diff_face_pairs[:min_size]
        self.pairs = np.concatenate((same_face_pairs, diff_face_pairs), axis=0)
        self.labels = np.concatenate((np.ones(len(same_face_pairs)), np.zeros(len(diff_face_pairs))), axis=0)

        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = self.pairs[idx][0]
        img2 = self.pairs[idx][1]

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
