"""Dataset-balanced, resumable distributed sampler (same interface as mimic-video's ResumableDistributedSampler).

Each epoch draws `len(dataset)` episode indices WITH replacement from `dataset.episode_weights()` (every dataset gets
equal total probability, spread over its episodes by duration), deterministic per (seed, epoch), split over ranks.
The time inside the episode is drawn by the dataset itself.
"""
from __future__ import annotations

import torch
from megatron.core import parallel_state
from torch.utils.data import Sampler


class WeightedResumableSampler(Sampler):
    def __init__(self, dataset, num_replicas: int, rank: int, seed: int = 0, dataset_weights: dict | None = None):
        self.weights = torch.as_tensor(dataset.episode_weights(dataset_weights), dtype=torch.double)
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.num_samples = len(dataset) // num_replicas
        self.epoch, self.start_iter = 0, 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.num_samples * self.num_replicas, replacement=True, generator=g)
        idx = idx[self.rank:: self.num_replicas][self.start_iter:]
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples - self.start_iter

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_iter(self, start_iter: int) -> None:
        assert start_iter < self.num_samples
        self.start_iter = start_iter


def get_weighted_sampler(dataset, dataset_weights: dict | None = None) -> WeightedResumableSampler:
    return WeightedResumableSampler(dataset, num_replicas=parallel_state.get_data_parallel_world_size(),
                                    rank=parallel_state.get_data_parallel_rank(), seed=0,
                                    dataset_weights=dataset_weights)
