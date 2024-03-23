from torch.nn import Module
from models.head import EmbeddingHead
from models.BiFormer.biformer import biformer_base


class BiFormerWrapper(Module):
    def __init__(self, embedding_size, num_classes):
        super(BiFormerWrapper, self).__init__()
        self.backbone = biformer_base()
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
