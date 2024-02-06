import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class OpenCLIPWrapper(nn.Module):
    def __init__(self, device):
        super(OpenCLIPWrapper, self).__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, imgs, txts):
        if len(imgs) != len(txts):
            raise ValueError("Number of images and text descriptions must be the same.")

        preprocessed_data = self.processor(text=txts, images=imgs, return_tensors="pt", padding=True).to(self.device)
        return self.clip_model(**preprocessed_data)