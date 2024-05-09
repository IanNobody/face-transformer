# Author: Šimon Strýček
# Description: Wrapper for Swin Transformer model.
# Year: 2024

from models.head import EmbeddingHead
from torchvision.models import swin_t
from models.wrappers.wrapper import Wrapper


class SWINWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(SWINWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = swin_t(pretrained=True)
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
