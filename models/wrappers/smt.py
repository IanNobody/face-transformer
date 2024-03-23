from torch.nn import Module
from models.head import EmbeddingHead
from models.SMT.smt import SMT


class SMTWrapper(Module):
    def __init__(self, embedding_size, num_classes):
        super(SMTWrapper, self).__init__()
        self.backbone = SMT(num_classes=num_classes)
        self.backbone.embed_fc = EmbeddingHead(self.backbone.embed_fc.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)["embedding"]
