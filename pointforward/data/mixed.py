import torch
from torch.utils.data import ConcatDataset, DistributedSampler


class MixedDataset(ConcatDataset):
    """Independent scene identities and equal dataset sampling probabilities."""

    def __init__(self, datasets, names):
        super().__init__(datasets)
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique")
        self.names = names
        self.scenes = [
            {**scene, "name": f"{name}/{scene['name']}"}
            for name, dataset in zip(names, datasets) for scene in dataset.scenes
        ]
        self.sample_weights = torch.cat([
            torch.full((len(dataset),), 1.0 / len(dataset), dtype=torch.double)
            for dataset in datasets
        ])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        item = super().__getitem__(index)
        source = next(position for position, end in enumerate(self.cumulative_sizes) if index < end)
        name = self.names[source]
        return {**item, "dataset": name, "scene": f"{name}/{item['scene']}"}


class MixedDistributedSampler(DistributedSampler):
    """Shard a reproducible balanced draw across torchrun ranks."""

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.dataset.sample_weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])