from torch.utils.data import Dataset
import torch
import cv2
from os.path import join
from os.path import isfile
import open_clip
from PIL import Image
import torch
import numpy as np

class VGGFaceDataset(Dataset):
    def __init__(self, img_dir, meta_dir, transform=None, augmentation=None, device=None):
        self.img_root = img_dir

        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')
        _, transform, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')
        self.transform = transform

        with open(join(meta_dir, "train_list_aligned.txt")) as f:
            self.img = [line for line in f.read().splitlines() if isfile(join(img_dir, line))]

        self.cls = [img.split('/')[-2] for img in self.img]
        self.cls_dict = {cls: idx for idx, cls in enumerate(set(self.cls))}

        with open(join(meta_dir, "identity_meta.csv")) as f:
            lines = f.read().splitlines()[1:]
            self.gender = {line.split(", ")[0]: (1.0 if line.split(", ")[-1] == 'm' else 0.0) for line in lines}

        self.hair = self._load_binary_meta_to_class([
            "02-Black_Hair.txt", "03-Brown_Hair.txt", "04-Gray_Hair.txt", "05-Blond_Hair.txt"
        ], meta_dir)
        self.glasses = self._load_binary_meta_to_class(["09-Eyeglasses.txt", "10-Sunglasses.txt"], meta_dir)
        self.mustache = self._load_binary_meta_to_class(["07-Mustache_or_Beard.txt"], meta_dir)
        self.hat = self._load_binary_meta_to_class(["08-Wearing_Hat.txt"], meta_dir)
        self.open_mouth = self._load_binary_meta_to_class(["11-Mouth_Open.txt"], meta_dir)
        self.long_hair = self._load_binary_meta_to_class(["06-Long_Hair.txt"], meta_dir)

        # self.transform = transform
        self.augmentation = augmentation
        self.device = device

    def _load_binary_meta_to_class(self, meta_files, meta_dir):
        if len(meta_files) > 1:
            meta = {cls: int(0) for cls in self.img}
        else:
            meta = {img: float(0) for img in self.img}

        for meta_id, meta_file in enumerate(meta_files):
            with open(join(meta_dir, meta_file)) as f:
                lines = f.read().splitlines()[1:]
                for line in lines:
                    raw_meta = line.split()
                    if raw_meta[1] == "1":
                        if len(meta_files) > 1:
                            meta[raw_meta[0]] = int(meta_id + 1)
                        else:
                            meta[raw_meta[0]] = float(1)

        return meta

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.img[idx]
        img_path = join(self.img_root, img_id)
        # image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = Image.open(img_path)

        cls_name = self.cls[idx]
        ann = torch.tensor(self.cls_dict[cls_name])

        gender = torch.tensor(self.gender[cls_name])
        hair = torch.tensor(self.hair[img_id])
        glasses = torch.tensor(self.glasses[img_id])
        mustache = torch.tensor(self.mustache[img_id])
        hat = torch.tensor(self.hat[img_id])
        open_mouth = torch.tensor(self.open_mouth[img_id])
        long_hair = torch.tensor(self.long_hair[img_id])

        # if self.augmentation:
        #     image = self.augmentation(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        text = self._textual_description({'gender': gender, 'hair': hair, 'glasses': glasses, 'mustache': mustache,
                                          'hat': hat, 'open_mouth': open_mouth, 'long_hair': long_hair})
        text = self.tokenizer(text)[0]

        sample = {'image': image, 'textual_prompt': text, "annotation": {'class': ann, 'gender': gender, 'hair': hair, 'glasses': glasses, 'mustache': mustache,
                  'hat': hat, 'open_mouth': open_mouth, 'long_hair': long_hair, 'id': img_id }}
        return sample

    def _textual_description(self, metadata):
        if metadata['gender'] == 1:
            text_prompt = "A man"
        else:
            text_prompt = "A woman"

        if (metadata['long_hair'] != 0 or metadata['hair'] != 0 or metadata['glasses'] != 0 or
                metadata['mustache'] != 0 or metadata['hat'] != 0 or metadata['open_mouth'] != 0):
            text_prompt += " with"

        if metadata['long_hair'] == 1:
            if metadata['hair'] == 0:
                text_prompt += " long hair"
            else:
                text_prompt += " long"

        if metadata['hair'] == 1:
            text_prompt += " black hair"
        elif metadata['hair'] == 2:
            text_prompt += " gray hair (or possibly monochrome image)"
        elif metadata['hair'] == 3:
            text_prompt += " brown hair"
        elif metadata['hair'] == 4:
            text_prompt += " blond hair"

        if len(text_prompt) > 12:
            text_prompt += ","

        if metadata['glasses'] == 1:
            text_prompt += " eyeglasses"

        if len(text_prompt) > 12:
            text_prompt += ","

        elif metadata['glasses'] == 2:
            text_prompt += " sunglasses"

        if len(text_prompt) > 12:
            text_prompt += ","

        if metadata['mustache'] == 1:
            text_prompt += " mustache or beard"

        if len(text_prompt) > 12:
            text_prompt += ","

        if metadata['hat'] == 1:
            text_prompt += " hat"

        if len(text_prompt) > 12:
            text_prompt += " and "

        if metadata['open_mouth'] == 1:
            text_prompt += " open mouth"

        text_prompt += "."
        return text_prompt

    def num_of_classes(self):
        return len(self.cls_dict)
