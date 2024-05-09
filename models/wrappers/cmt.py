# Author: Šimon Strýček
# Description: Wrapper for the CMT model.
# Year: 2024

from models.head import EmbeddingHead
from models.CMT.cmt import cmt_b
from models.wrappers.wrapper import Wrapper


class CMTWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(CMTWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = cmt_b(num_classes=num_classes)
        self.backbone.embed_fc = EmbeddingHead(self.backbone.embed_fc.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)["embedding"]
