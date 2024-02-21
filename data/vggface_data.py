from torch.utils.data import Dataset
import torch
import cv2
from os.path import join
from os.path import isfile

class VGGFaceDataset(Dataset):
    def __init__(self, img_dir, meta_dir, transform=None, augmentation=None, device=None):
        self.img_root = img_dir

        with open(join(meta_dir, "train_list_aligned.txt")) as f:
            self.img = [line for line in f.read().splitlines() if isfile(join(img_dir, line))]

        self.cls = [img.split('/')[-2] for img in self.img]
        self.cls_dict = {cls: idx for idx, cls in enumerate(set(self.cls))}

        with open(join(meta_dir, "identity_meta.csv")) as f:
            lines = f.read().splitlines()[1:]
            self.gender = {line.split(", ")[0]: (1 if line.split(", ")[-1] == 'm' else 0) for line in lines}

        self.hair = {cls: 0 for cls in self.img}

        for hair_id, hair_meta_file in enumerate(["02-Black_Hair.txt", "03-Brown_Hair.txt", "04-Gray_Hair.txt", "05-Blond_Hair.txt"]):
            with open(join(meta_dir, hair_meta_file)) as f:
                lines = f.read().splitlines()[1:]
                for line in lines:
                    meta = line.split()
                    if meta[1] == "1":
                        self.hair[meta[0]] = hair_id + 1

        self.transform = transform
        self.augmentation = augmentation
        self.device = device

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.img[idx]
        img_path = join(self.img_root, img_id)
        image = cv2.imread(img_path)

        cls_name = self.cls[idx]
        ann = torch.tensor(self.cls_dict[cls_name])
        gender = torch.tensor(self.gender[cls_name])
        hair = torch.tensor(self.hair[img_id])

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'class': ann, 'gender': gender, 'hair': hair}
        return sample

    def num_of_classes(self):
        return max(self.ann) + 1
