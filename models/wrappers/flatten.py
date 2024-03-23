from torch.nn import Module
from models.head import EmbeddingHead
from models.Flatten_T.flatten_swin import FLattenSwinTransformer


class FLattenWrapper(Module):
    def __init__(self, embedding_size, num_classes):
        super(FLattenWrapper, self).__init__()
        self.backbone = FLattenSwinTransformer(num_classes=num_classes)
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
