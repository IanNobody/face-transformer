from torch.nn import Module
from models.head import EmbeddingHead
from models.DAT.dat import DAT
from models.wrappers.wrapper import Wrapper


class DATWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(DATWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = DAT(num_classes=num_classes)
        self.backbone.cls_head = EmbeddingHead(self.backbone.cls_head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
