import torch.nn as nn
import open_clip

class MultitaskOpenCLIP(nn.Module):
    def __init__(self, device, num_classes):
        super(MultitaskOpenCLIP, self).__init__()

        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')

        self.embed_fc = nn.Linear(512, 512)
        self.class_fc = nn.Linear(512, num_classes)
        self.gender_fc = nn.Linear(512, 1)
        self.hair_fc = nn.Linear(512, 4)
        self.glasses_fc = nn.Linear(512, 2)
        self.mustache_fc = nn.Linear(512, 1)
        self.hat_fc = nn.Linear(512, 1)
        self.open_mouth_fc = nn.Linear(512, 1)
        self.long_hair_fc = nn.Linear(512, 1)

    def forward(self, x, textual_prompt):
        y = self.model(text=textual_prompt, image=x)[0]

        embed = self.embed_fc(y)
        cls = self.class_fc(y)

        gender = self.gender_fc(y).squeeze()
        hair = self.hair_fc(y)
        glasses = self.glasses_fc(y)
        mustache = self.mustache_fc(y).squeeze()
        hat = self.hat_fc(y).squeeze()
        open_mouth = self.open_mouth_fc(y).squeeze()
        long_hair = self.long_hair_fc(y).squeeze()

        return {"embedding": embed, "class": cls, "gender": gender, "hair": hair, "glasses": glasses, "mustache": mustache,
                "hat": hat, "open_mouth": open_mouth, "long_hair": long_hair,  "gender": gender}
