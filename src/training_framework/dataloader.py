import torch
from torch.utils.data import Sampler


class InfiniteSampler(Sampler):
    def __init__(self, n_samples):
        super().__init__()
        self._n_samples = n_samples

    def __iter__(self):
        while True:
            yield from torch.randperm(self._n_samples).tolist()