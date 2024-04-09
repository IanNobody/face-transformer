from torch.nn import Module
from models.head import EmbeddingHead
from models.NoisyViT.noisy_vit import vit_b
from models.wrappers.wrapper import Wrapper


class NoisyViTWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(NoisyViTWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = vit_b()
        self.backbone.head = EmbeddingHead(self.backbone.head.in_features, embedding_size, num_classes)

    def forward(self, x):
        return self.backbone(x)
