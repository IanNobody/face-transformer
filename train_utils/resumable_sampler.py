from torch.utils.data import Sampler
import torch


class ResumableRandomSampler(Sampler):
    def __init__(self, data_source, config):
        super().__init__(data_source)
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(config.sampler_seed)

        self.perm_index = torch.tensor(0).share_memory_()
        self.perm = torch.randperm(self.num_samples, generator=self.generator).share_memory_()

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}

    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])