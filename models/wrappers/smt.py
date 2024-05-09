# Author: Šimon Strýček
# Description: Wrapper for the SMT model.
# Year: 2024

from models.head import EmbeddingHead
from models.SMT.smt import SMT
from models.wrappers.wrapper import Wrapper


class SMTWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(SMTWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = SMT(num_classes=num_classes)
        self.backbone.embed_fc = EmbeddingHead(self.backbone.embed_fc.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)["embedding"]
