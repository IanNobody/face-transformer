import os
import torch
from torch.utils.data import Dataset
import mxnet as mx
from PIL import Image
import numpy as np
import open_clip
import random


class MS1M_Dataset(Dataset):
    def __init__(self, rec_path, augmentation=None):
        super(MS1M_Dataset).__init__()
        self.rec_path = rec_path
        self.idx_path = rec_path + 'train.idx'
        self.rec_file = rec_path + 'train.rec'
        self.augmentation = augmentation
        _, transform, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')
        self.transform = transform
        self.recordio = mx.recordio.MXIndexedRecordIO(self.idx_path, self.rec_file, 'r')
        self.keys = list(self.recordio.keys)

    # def __len__(self):
    #     return len(self.keys)

    def __len__(self):
        return 5822653

    def num_of_classes(self):
        return 100000

    def __getitem__(self, idx):
        while True:
            try:
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                    record = self.recordio.read_idx(self.keys[idx])
                    header, img = mx.recordio.unpack_img(record)
                    break
            except:
                print("ERROR: Failed to read record at index", idx)
                idx = random.randint(0, len(self))
                print("Rollback to index", idx)
                continue

        img = img[:, :, ::-1]

        # Assuming single label per image, adjust if necessary
        label = int(header.label) if isinstance(header.label, float) else header.label[0]

        if self.transform:
            img = self.augmentation(image=img)["image"]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return {"image": img, "annotation": {"class": label}}