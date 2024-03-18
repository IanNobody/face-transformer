import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class PretrainedOpenCLIPWrapper(nn.Module):
    def __init__(self, pretrained_original_openclip, num_classes):
        super(PretrainedOpenCLIPWrapper, self).__init__()
        self.backbone = pretrained_original_openclip.model
        self.embed_fc = pretrained_original_openclip.embed_fc
        self.gender_fc = pretrained_original_openclip.gender_fc
        self.class_fc = nn.Linear(512, num_classes + 1)
        self.hair_fc = nn.Linear(512, 6)
        self.glasses_fc = pretrained_original_openclip.glasses_fc
        self.mustache_fc = pretrained_original_openclip.mustache_fc
        self.hat_fc = pretrained_original_openclip.hat_fc
        self.open_mouth_fc = pretrained_original_openclip.open_mouth_fc

    def forward(self, x, textual_prompt):
        y = self.backbone(text=textual_prompt, image=x)[0]
        embed = self.embed_fc(y)
        cls = self.class_fc(y)

        gender = self.gender_fc(y).squeeze()
        hair = self.hair_fc(y)
        glasses = self.glasses_fc(y)
        mustache = self.mustache_fc(y).squeeze()
        hat = self.hat_fc(y).squeeze()
        open_mouth = self.open_mouth_fc(y).squeeze()

        return {"embedding": embed, "class": cls, "hair": hair, "glasses": glasses,
                "mustache": mustache, "hat": hat, "open_mouth": open_mouth, "gender": gender}
