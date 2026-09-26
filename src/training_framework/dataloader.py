from typing import Any, Dict, Optional
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


#: Bumped when the shape of a saved sampler position changes. Version 2
#: records how far the epoch got across all ranks rather than how far one
#: rank got, which is what lets a run resume on a different world size. A
#: state without the key is version 1.
SAMPLER_STATE_VERSION = 2


def samples_per_rank(
        num_samples: int,
        world_size: int,
        drop_last: bool = False,
) -> int:
    """How many indices each rank receives in one epoch."""
    if drop_last:
        return num_samples // world_size
    return (num_samples + world_size - 1) // world_size


def epoch_size(
        num_samples: int,
        world_size: int,
        drop_last: bool = False,
) -> int:
    """How many indices one epoch delivers across every rank together."""
    return samples_per_rank(num_samples, world_size, drop_last) * world_size


def consumed_in_epoch(state: Dict[str, Any]) -> int:
    """How many samples of the epoch were consumed, across all ranks."""
    if "consumed_in_epoch" in state:
        return state["consumed_in_epoch"]
    # Version 1 recorded one rank's position and the world size it was taken
    # on. Ranks advance in lockstep, so every rank had consumed the same
    # number and the total is simply their product.
    return state.get("index_within_epoch", 0) * (state.get("world_size") or 1)


def _normalized_position(state: Dict[str, Any]) -> tuple[int, int]:
    """Fold whole epochs out of a saved count, in the geometry it was taken in.

    The count is of positions in a *padded* epoch, and how much padding an
    epoch carries depends on the world size that did the counting. A sampler
    saved just after yielding an epoch's last item has not incremented its
    epoch yet -- a generator is suspended at the yield, not past it -- so the
    count can be a whole epoch's worth. Folded out here against the epoch it
    belongs to, that reads as a completed epoch; left in, it would look like
    an overflow of the resuming epoch and push every rank past its start.
    """
    consumed = consumed_in_epoch(state)
    epoch = state.get("epoch", 0)

    world_size = state.get("world_size")
    num_samples = state.get("num_samples")
    if not world_size or not num_samples:
        return epoch, consumed

    size = epoch_size(num_samples, world_size, state.get("drop_last", False))
    if size <= 0:
        return epoch, consumed

    completed_epochs, consumed = divmod(consumed, size)
    return epoch + completed_epochs, consumed



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
        self.num_samples_per_rank = samples_per_rank(
            self.num_samples,
            self.world_size,
            self.drop_last,
        )
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
        """Serializes the current state of the sampler for checkpointing.

        The position is recorded as the number of samples the epoch has
        delivered across all ranks. Ranks advance in lockstep, so that count
        means the same thing whatever the world size -- which is what lets a
        run resume on a different number of processes.
        """
        return {
            "state_version": SAMPLER_STATE_VERSION,
            "epoch": self.epoch,
            "consumed_in_epoch": self.index_within_epoch * self.world_size,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            # A resumed run takes its topology from the launch, never from
            # here. `world_size` is still needed to read `consumed_in_epoch`,
            # which counts positions in an epoch whose padding that world
            # size decided; `rank` is recorded for diagnostics.
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a saved position, rebased onto this sampler's topology.

        A saved position belongs to the world size that wrote it. This
        sampler's own rank and world size, settled when it was built, are
        what the run uses now, so the position is rebased onto them rather
        than adopted.
        """
        self.seed = state.get("seed", self.seed)
        self.num_samples = state.get("num_samples", self.num_samples)
        self.shuffle = state.get("shuffle", self.shuffle)
        self.drop_last = state.get("drop_last", self.drop_last)

        self._compute_sample_counts()

        self.epoch, self.index_within_epoch = self._rebased_position(state)

    def _rebased_position(self, state: Dict[str, Any]) -> tuple[int, int]:
        """Place a saved position in this sampler's epoch.

        Rounding is upward, so a resumed run never delivers a sample it has
        already delivered; instead it may skip up to `world_size - 1` of the
        epoch's remaining samples. The permutation depends only on the seed
        and the epoch, so every position below `num_samples` names the same
        sample under any world size -- only the padding differs, which
        confines the imprecision to the tail of an epoch.
        """
        if self.num_samples_per_rank <= 0:
            raise ValueError(
                f"A dataset of {self.num_samples} sample(s) cannot be split "
                f"across {self.world_size} ranks"
            )

        epoch, consumed = _normalized_position(state)
        index = -(-consumed // self.world_size)
        # A smaller world size can push the rebased index past the end of
        # its shorter epoch; carrying keeps the sampler from slicing an
        # epoch to nothing.
        carried_epochs, index = divmod(index, self.num_samples_per_rank)
        return epoch + carried_epochs, index