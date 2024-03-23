import os
import tarfile
from zipfile import ZipFile
from torch.utils.data import Dataset
import cv2
from os.path import join
from os.path import isfile
import open_clip
import torch
import numpy as np
import PIL.Image as Image

class CelebADataset(Dataset):
    def __init__(self, img_dir, meta_dir, augmentation=None, device=None):
        self.img_root = img_dir

        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')
        _, transform, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')
        self.transform = transform

        self.img = []
        self.cls = []
        with open(join(meta_dir, "identity_CelebA.txt")) as f:
            a = f.read().splitlines()
            for idx, line in enumerate(a):
                filename, class_id = line.split()
                filename = filename.replace("jpg", "png")
                if isfile(join(img_dir, filename)):
                    self.img.append(filename)
                    self.cls.append(class_id)

        self.class_dict = {cls: idx for idx, cls in enumerate(set(self.cls))}

        self.attr = {}
        with open(join(meta_dir, "list_attr_celeba.txt")) as f:
            a = f.read().splitlines()[1:]
            for idx, line in enumerate(a):
                filename, *attr = line.split()
                filename = filename.replace("jpg", "png")
                if filename in self.img:
                    hair = [attr[4], attr[8], attr[9], attr[11], attr[17]]
                    hair = hair.index("1") + 1 if "1" in hair else 0
                    glasses = 0 if attr[15] == "-1" else 1
                    mustache = 0 if attr[24] == "1" else 1
                    gender = 1 if attr[20] == "1" else 0
                    mouth_open = 1 if attr[21] == "1" else 0
                    hat = 1 if attr[35] == "1" else 0
                    self.attr[filename] = [hair, glasses, mustache, gender, mouth_open, hat]

        self.augmentation = augmentation
        self.device = device

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.img[idx]
        img_path = join(self.img_root, img_id)
        image = np.array(Image.open(img_path))

        cls_name = self.cls[idx]
        ann = torch.tensor(self.class_dict[cls_name])

        hair, glasses, mustache, gender, mouth_open, hat = self.attr[img_id]
        gender = torch.tensor(gender)
        hair = torch.tensor(hair)
        glasses = torch.tensor(glasses)
        mustache = torch.tensor(mustache)
        hat = torch.tensor(hat)
        open_mouth = torch.tensor(mouth_open)

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        text = self._textual_description({'gender': gender, 'hair': hair, 'glasses': glasses, 'mustache': mustache,
                                          'hat': hat, 'open_mouth': open_mouth})
        text = self.tokenizer(text)[0]

        sample = {'data': {'image': image, 'textual_prompt': text},
                  "annotation": {'class': ann, 'gender': gender, 'hair': hair, 'glasses': glasses, 'mustache': mustache,
                                 'hat': hat, 'open_mouth': open_mouth, 'id': img_id}}
        return sample

    def _textual_description(self, metadata):
        if metadata['gender'] == 1:
            text_prompt = "A man"
        else:
            text_prompt = "A woman"

        if (metadata['hair'] != 0 or metadata['glasses'] != 0 or
                metadata['mustache'] != 0 or metadata['hat'] != 0 or metadata['open_mouth'] != 0):
            text_prompt += " with"

        if metadata['hair'] == 1:
            text_prompt += " no hair"
        elif metadata['hair'] == 2:
            text_prompt += " black hair"
        elif metadata['hair'] == 3:
            text_prompt += " blonde hair"
        elif metadata['hair'] == 4:
            text_prompt += " brown hair"
        elif metadata['hair'] == 5:
            text_prompt += " gray hair (or it is just a monochrome image)"

        if len(text_prompt) > 5:
            text_prompt += ","

        if metadata['glasses'] == 1:
            text_prompt += " eyeglasses"

        if len(text_prompt) > 5:
            text_prompt += ","

        if len(text_prompt) > 5:
            text_prompt += ","

        if metadata['mustache'] == 1:
            text_prompt += " mustache"

        if len(text_prompt) > 5:
            text_prompt += ","

        if metadata['hat'] == 1:
            text_prompt += " hat"

        if len(text_prompt) > 5:
            text_prompt += " and "

        if metadata['open_mouth'] == 1:
            text_prompt += " open mouth"

        text_prompt += "."
        return text_prompt

    def num_classes(self):
        return len(self.class_dict)