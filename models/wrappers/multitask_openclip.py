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
        self.task_probabilities = {
            "class_fc": 0.3, "gender_fc": 0.2, "hair_fc": 0.15, "glasses_fc": 0.05, "mustache_fc": 0.15,
            "hat_fc": 0.05, "open_mouth_fc": 0.05, "long_hair_fc": 0.05
        }
        # self.task_probabilities = {
        #     "class_fc": 0.6, "gender_fc": 0.25, "hair_fc": 0.05, "glasses_fc": 0.05, "mustache_fc": 0.05,
        #     "hat_fc": 0.0, "open_mouth_fc": 0.0, "long_hair_fc": 0.0
        # }

    def forward(self, x, text_prompt):
        y = self.backbone(text=text_prompt, image=x)

        rec = self.recognition_head(y[0])

        embed = rec["embedding"]
        gender = self.gender_fc(embed).squeeze()
        hair = self.hair_fc(embed)
        glasses = self.glasses_fc(embed)
        mustache = self.mustache_fc(embed).squeeze()
        hat = self.hat_fc(embed).squeeze()
        open_mouth = self.open_mouth_fc(embed).squeeze()
        long_hair = self.long_hair_fc(embed).squeeze()

        return {
            "embedding": embed, "class": rec["class"], "gender": gender, "hair": hair,
            "glasses": glasses, "mustache": mustache, "hat": hat, "open_mouth": open_mouth,
            "long_hair": long_hair, "raw": y
        }

    def _switch_task_by_layer_name(self, layer_name, device):
        self.active_task_layer = layer_name

        for name, param in self.named_parameters():
            if name.startswith(layer_name):
                param.requires_grad_(True)
            elif self._is_task_layer(name):
                param.requires_grad_(False)

    def _is_task_layer(self, name):
        return True if "class_fc" in name else any(name.startswith(task) for task in self.task_probabilities.keys())

    def switch_random_task(self, device):
        rand = self.task_random.random()
        for task_name in self.task_probabilities.keys():
            if rand > self.task_probabilities[task_name]:
                rand -= self.task_probabilities[task_name]
            else:
                self._switch_task_by_layer_name(task_name, device)
                break

    def load_backbone_weights(self, path):
        if path is not None:
            weights = torch.load(path, map_location='cpu')["state_dict"]
            backbone_weights = {k: v for k, v in weights.items() if k.startswith('model.backbone')}
            self.backbone.load_state_dict(backbone_weights, strict=False)

