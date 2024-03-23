from torch.nn import Module
from models.head import EmbeddingHead
from torchvision.models import resnet50


class ResNet50Wrapper(Module):
    def __init__(self, embedding_size, num_classes):
        super(ResNet50Wrapper, self).__init__()
        self.backbone = resnet50()
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
