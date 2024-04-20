from transformers import CLIPModel
import torch.nn as nn
import torch
from models.wrappers.wrapper import Wrapper

class OldOpenCLIPWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(OldOpenCLIPWrapper, self).__init__(embedding_size, num_classes)
        self.clip_vision = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        self.fc = nn.Linear(768, embedding_size)
        # self.class_fc = nn.Linear(768, num_classes)

    def forward(self, x):
        y = self.clip_vision(x).pooler_output
        embed = self.fc(y)
        # cls = self.class_fc(y)
        # return {"embedding": embed, "class": cls}
        return {"embedding": embed}

    def load_backbone_weights(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path, map_location="cpu")["model_weights"])