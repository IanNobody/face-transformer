from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from os.path import join, isdir
from os import listdir
import os


class JoinedDataset(Dataset):
    def __init__(self, target_dir, ref_dir, transform=None, device=None):
        self.target_classes = [dir for dir in listdir(target_dir) if isdir(join(target_dir, dir))]
        self.ref_classes = [dir for dir in listdir(ref_dir) if isdir(join(ref_dir, dir))]

        self.target_path = target_dir
        self.target_files = [[join(target, file) for file in listdir(join(target_dir, target)) if file.endswith(".jpg") or file.endswith(".png")][0] for target in self.target_classes]
        self.ref_path = ref_dir
        self.ref_files = [[join(ref, file) for file in listdir(join(ref_dir, ref)) if file.endswith(".jpg") or file.endswith(".png")][0] for ref in self.ref_classes]

        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.target_files) * len(self.ref_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_idx = idx % len(self.ref_files)
        target_idx = idx // len(self.ref_files)

        ref_path = join(self.ref_path, self.ref_files[ref_idx])
        target_path = join(self.target_path, self.target_files[target_idx])

        ref_image = cv2.imread(ref_path).astype(np.float32)
        target_image = cv2.imread(target_path).astype(np.float32)
        ref_class = self.ref_classes[ref_idx]
        target_class = self.target_classes[target_idx]

        if self.transform:
            ref_image = self.transform(ref_image)
            target_image = self.transform(target_image)

        if self.device:
            ref_image = ref_image.to(self.device)
            target_image = target_image.to(self.device)

        sample = {'ref_image': ref_image, 'target_image': target_image, 'ref_class': ref_class, 'target_class': target_class}
        return sample

    @staticmethod
    def collate_fn(batch):
        ref_imgs = torch.stack([d["ref_image"] for d in batch])
        target_imgs = torch.stack([d["target_image"] for d in batch])
        ref_classes = [d["ref_class"] for d in batch]
        target_classes = [d["target_class"] for d in batch]
        return {"ref_image": ref_imgs, "target_image": target_imgs, "ref_class": ref_classes, "target_class": target_classes}

