from torch.nn import Module
from models.head import EmbeddingHead
from torchvision.models import swin_b


class SWINWrapper(Module):
    def __init__(self, embedding_size, num_classes):
        super(SWINWrapper, self).__init__()
        self.backbone = swin_b()
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
