from torch.nn import Module
from models.head import EmbeddingHead
from models.CMT.cmt import cmt_b


class CMTWrapper(Module):
    def __init__(self, embedding_size, num_classes):
        super(CMTWrapper, self).__init__()
        self.backbone = cmt_b(num_classes=num_classes)
        self.backbone.embed_fc = EmbeddingHead(self.backbone.embed_fc.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)["embedding"]
