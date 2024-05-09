# Author: Šimon Strýček
# Description: Wrapper for the ResNet50 model.
# Year: 2024

from models.head import EmbeddingHead
from torchvision.models import resnet50
from models.wrappers.wrapper import Wrapper


class ResNet50Wrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(ResNet50Wrapper, self).__init__(embedding_size, num_classes)
        self.backbone = resnet50()
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
