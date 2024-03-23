import torch.nn as nn
from open_clip import create_model
from models.head import EmbeddingHead


class MultitaskOpenCLIPWrapper(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(MultitaskOpenCLIPWrapper, self).__init__()
        self.backbone = create_model('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')

        self.recognition_head = EmbeddingHead(512, embedding_size, num_classes)
        self.gender_fc = nn.Linear(512, 1)
        self.hair_fc = nn.Linear(512, 5)
        self.glasses_fc = nn.Linear(512, 3)
        self.mustache_fc = nn.Linear(512, 1)
        self.hat_fc = nn.Linear(512, 1)
        self.open_mouth_fc = nn.Linear(512, 1)
        self.long_hair_fc = nn.Linear(512, 1)

    def forward(self, x, text_prompt):
        y = self.model(text=text_prompt, image=x)[0]

        rec = self.recognition_head(y)
        gender = self.gender_fc(y).squeeze()
        hair = self.hair_fc(y)
        glasses = self.glasses_fc(y)
        mustache = self.mustache_fc(y).squeeze()
        hat = self.hat_fc(y).squeeze()
        open_mouth = self.open_mouth_fc(y).squeeze()
        long_hair = self.long_hair_fc(y).squeeze()

        return {"embedding": rec["embedding"], "class": rec["class"], "gender": gender, "hair": hair,
                "glasses": glasses, "mustache": mustache, "hat": hat, "open_mouth": open_mouth,
                "long_hair": long_hair,  "gender": gender}
