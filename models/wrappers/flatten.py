# Author: Šimon Strýček
# Description: Wrapper for the FLatten Swin transformer model
# Year: 2024

from models.head import EmbeddingHead
from models.Flatten_T.flatten_swin import FLattenSwinTransformer
from models.wrappers.wrapper import Wrapper


class FLattenWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(FLattenWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = FLattenSwinTransformer(num_classes=num_classes)
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
