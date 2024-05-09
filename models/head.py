# Author: Šimon Strýček
# Description: Implementation of the embedding/classification head.
# Year: 2024

from torch.nn import Module, Linear


class EmbeddingHead(Module):
    def __init__(self, input_size, embedding_size, num_classes):
        super(EmbeddingHead, self).__init__()
        self.embed_fc = Linear(input_size, embedding_size)
        self.class_fc = Linear(embedding_size, num_classes + 1)

    def forward(self, x):
        embed = self.embed_fc(x)
        cls = self.class_fc(embed)
        return {"embedding": embed, "class": cls}