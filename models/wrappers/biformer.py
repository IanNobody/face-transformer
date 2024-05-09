# Author: Šimon Strýček
# Description: Wrapper for the BiFormer model.
# Year: 2024

from models.head import EmbeddingHead
from models.BiFormer.biformer import biformer_base
from models.wrappers.wrapper import Wrapper


class BiFormerWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(BiFormerWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = biformer_base()
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
