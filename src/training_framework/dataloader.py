from typing import Any, Dict, Optional
import torch
from torch.utils.data import Sampler
import torch.distributed as dist



class InfiniteSampler(Sampler):
    def __init__(self, n_samples):
        super().__init__()
        self._n_samples = n_samples

    def __iter__(self):
        while True:
            yield from torch.randperm(self._n_samples).tolist()


class DistributedInfiniteSampler(Sampler):
    """
    An infinite distributed sampler supporting state serialization (checkpointing)
    and deserialization via `get_state` and `set_state`.
    """

    def __init__(
        self,
        num_samples: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Internal state tracking for serialization
        self.epoch: int = 0
        self.index_within_epoch: int = 0

        self._resolve_distributed_context()
        self._compute_sample_counts()

    def _resolve_distributed_context(self) -> None:
        """Resolves rank and world_size from PyTorch distributed if not explicitly passed."""
        if self.rank is None:
            self.rank = (
                dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0
            )
        if self.world_size is None:
            self.world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )

    def _compute_sample_counts(self) -> None:
        """Calculates total indices and per-rank slice sizes."""
        if self.drop_last and self.num_samples % self.world_size != 0:
            self.num_samples_per_rank = self.num_samples // self.world_size
            self.total_size = self.num_samples_per_rank * self.world_size
        else:
            self.num_samples_per_rank = (
                self.num_samples + self.world_size - 1
            ) // self.world_size
            self.total_size = self.num_samples_per_rank * self.world_size

    def _generate_epoch_indices(self, epoch: int) -> list[int]:
        """Generates rank-specific indices for a given epoch pass."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size // len(indices)) + 1))[
                    :padding_size
                ]
        else:
            indices = indices[: self.total_size]

        # Return subsampled slice assigned to this rank
        return indices[self.rank : self.total_size : self.world_size]

    def __iter__(self):
        while True:
            rank_indices = self._generate_epoch_indices(self.epoch)

            # Resume from saved position within epoch if restoring from state
            start_idx = self.index_within_epoch
            self.index_within_epoch = 0  # Reset for subsequent epochs

            for idx in rank_indices[start_idx:]:
                self.index_within_epoch += 1
                yield idx

            # Epoch finished -> advance epoch and reset intra-epoch index counter
            self.epoch += 1
            self.index_within_epoch = 0

    def __len__(self) -> int:
        return self.num_samples_per_rank

    # -------------------------------------------------------------------------
    # State Serialization Methods
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Serializes the current state of the sampler for checkpointing."""
        return {
            "epoch": self.epoch,
            "index_within_epoch": self.index_within_epoch,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "rank": self.rank,
            "world_size": self.world_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores sampler state from a checkpoint dictionary."""
        self.epoch = state.get("epoch", 0)
        self.index_within_epoch = state.get("index_within_epoch", 0)
        self.seed = state.get("seed", self.seed)
        self.num_samples = state.get("num_samples", self.num_samples)
        self.rank = state.get("rank", self.rank)
        self.world_size = state.get("world_size", self.world_size)
        self.shuffle = state.get("shuffle", self.shuffle)
        self.drop_last = state.get("drop_last", self.drop_last)

        self._compute_sample_counts()