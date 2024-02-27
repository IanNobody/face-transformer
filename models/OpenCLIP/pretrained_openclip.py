import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class OpenCLIPWrapper(nn.Module):
    def __init__(self, device=None):
        super(OpenCLIPWrapper, self).__init__()
        self.device = device
        self.clip_vision = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        self.fc = nn.Linear(768, 512)
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, x):
        y = self.clip_vision(x).pooler_output
        y = self.fc(y)
        return y

    # def forward(self, imgs, txts):
    #     if len(imgs) != len(txts):
    #         raise ValueError("Number of images and text descriptions must be the same.")
    #
    #     preprocessed_data = self.processor(text=txts, images=imgs, return_tensors="pt", padding=True).to(self.device)
    #     return self.clip_model(**preprocessed_data)