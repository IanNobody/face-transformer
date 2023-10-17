from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
import numpy as np
import os
import tarfile
from zipfile import ZipFile


class CelebADataset(Dataset):
    def __init__(self, ann_path, img_dir, transform=None, device=None):
        raw_ann = pd.read_csv(ann_path, delim_whitespace=True, header=None)
        self.img_names = raw_ann[0]
        self.ann = raw_ann[1]
        self.img_dir = img_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path).astype(np.float32)
        ann = torch.tensor(self.ann[idx])

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.device:
            image = image.to(self.device)
            ann = ann.to(self.device)

        sample = {'image': image, 'entity': ann}
        return sample

    @staticmethod
    def _try_clean(extract_path):
        if os.path.isfile(extract_path):
            os.remove(extract_path)
        else:
            print("Error: Could not clean the archive after extracting.")

    @staticmethod
    def _extract_tgz_file(extract_path):
        tar = tarfile.open(extract_path)
        tar.extractall()
        tar.close()
        CelebADataset._try_clean(extract_path)

    @staticmethod
    def _extract_zip_file(extract_path):
        with ZipFile(extract_path) as z_file:
            z_file.extractall(extract_path[:-4])
        CelebADataset._try_clean(extract_path)

    @staticmethod
    def extract_archive(archive_path):
        if archive_path.endswith(".tgz"):
            CelebADataset._extract_tgz_file(archive_path)
        elif archive_path.endswith(".zip"):
            CelebADataset._extract_zip_file(archive_path)
        else:
            raise AssertionError("Only .tgz and .zip files are supported.")