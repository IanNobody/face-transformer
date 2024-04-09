import random
import torch
import torch.nn as nn
from models.wrappers.wrapper import Wrapper
from open_clip import create_model
from models.head import EmbeddingHead


class MultitaskOpenCLIPWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(MultitaskOpenCLIPWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = create_model('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')

        self.recognition_head = EmbeddingHead(512, embedding_size, num_classes)
        self.gender_fc = nn.Linear(512, 1)
        self.hair_fc = nn.Linear(512, 5)
        self.glasses_fc = nn.Linear(512, 3)
        self.mustache_fc = nn.Linear(512, 1)
        self.hat_fc = nn.Linear(512, 1)
        self.open_mouth_fc = nn.Linear(512, 1)
        self.long_hair_fc = nn.Linear(512, 1)

        self.task_random = random.Random(412)
        self.active_task_layer = "class_fc"
        # self.task_probabilities = {
        #     "class_fc": 0.3, "gender_fc": 0.2, "hair_fc": 0.15, "glasses_fc": 0.05, "mustache_fc": 0.15,
        #     "hat_fc": 0.05, "open_mouth_fc": 0.05, "long_hair_fc": 0.05
        # }
        self.task_probabilities = {
            "class_fc": 0.125, "gender_fc": 0.125, "hair_fc": 0.125, "glasses_fc": 0.125, "mustache_fc": 0.125,
            "hat_fc": 0.125, "open_mouth_fc": 0.125, "long_hair_fc": 0.125
        }

    def forward(self, x, text_prompt):
        y = self.backbone(text=text_prompt, image=x)[0]

        rec = self.recognition_head(y)
        gender = self.gender_fc(y).squeeze()
        hair = self.hair_fc(y)
        glasses = self.glasses_fc(y)
        mustache = self.mustache_fc(y).squeeze()
        hat = self.hat_fc(y).squeeze()
        open_mouth = self.open_mouth_fc(y).squeeze()
        long_hair = self.long_hair_fc(y).squeeze()

        return {
            "embedding": rec["embedding"], "class": rec["class"], "gender": gender, "hair": hair,
            "glasses": glasses, "mustache": mustache, "hat": hat, "open_mouth": open_mouth,
            "long_hair": long_hair
        }

    def _switch_task_by_layer_name(self, layer_name):
        self.active_task_layer = layer_name

        for name, param in self.named_parameters():
            if name.startswith(layer_name):
                param.requires_grad_(True)
                print(name, end=", ")
            else:
                param.requires_grad_(False)
                print("<", name, ">", end=", ")

    def switch_random_task(self):
        rand = self.task_random.random()
        for task_name in self.task_probabilities.keys():
            if rand > self.task_probabilities[task_name]:
                rand -= self.task_probabilities[task_name]
            else:
                print("Picked ", task_name)
                self._switch_task_by_layer_name(task_name)
                break

    def load_backbone_weights(self, path):
        weights = torch.load(path, map_location='cpu')["state_dict"]
        backbone_weights = {k: v for k, v in weights.items() if k.startswith('model.backbone')}
        self.backbone.load_state_dict(backbone_weights, strict=False)

