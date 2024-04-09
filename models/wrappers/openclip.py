from torch.nn import Module
from models.head import EmbeddingHead
from open_clip import create_model
from models.wrappers.wrapper import Wrapper


class OpenCLIPVisionWrapper(Wrapper):
    def __init__(self, embedding_size, num_classes):
        super(OpenCLIPVisionWrapper, self).__init__(embedding_size, num_classes)
        self.backbone = create_model('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K').visual
        self.head = EmbeddingHead(512, embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
