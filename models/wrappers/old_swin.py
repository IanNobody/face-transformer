from torch.nn import Module
from models.head import EmbeddingHead
from torchvision.models import swin_t
from models.wrappers.wrapper import Wrapper
import torch.nn as nn
import torch


class OldSWINWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(OldSWINWrapper, self).__init__(embedding_size, num_classes)
        self.swin = swin_t(pretrained=True)
        self.swin.head = nn.Linear(self.swin.head.in_features, embedding_size)

    def forward(self, x):
        return {"embedding": self.swin(x)}

    def load_backbone_weights(self, path):
        if path is not None:
            self.swin.load_state_dict(torch.load(path, map_location="cpu")["model_weights"])
