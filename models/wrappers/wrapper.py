import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, embedding_size, num_of_classes):
        super(Wrapper, self).__init__()
        self.embedding_size = embedding_size
        self.num_of_classes = num_of_classes
        self.backbone = None

    def forward(self, x):
        raise NotImplementedError

    def load_backbone_weights(self, path):
        self.backbone.load_state_dict(path, strict=False)