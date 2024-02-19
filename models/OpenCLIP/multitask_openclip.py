import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class MultitaskOpenCLIP(nn.Module):
    age_bins = 11
    bin_size = 10

    def __init__(self, device):
        super(MultitaskOpenCLIP, self).__init__()

        self.device = device
        self.clip_vision = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model

        self.embed_fc = nn.Linear(768, 512)
        self.gender_fc = nn.Linear(768, 1)
        self.age_fc = nn.Linear(768, self.age_bins)

    def forward(self, x):
        y = self.clip_vision(x).pooler_output
        embed = self.fc(y)
        gender = self.gender_fc(y)
        age = self.age_fc(y)
        return {"embedding": embed, "gender": gender, "age": age}
