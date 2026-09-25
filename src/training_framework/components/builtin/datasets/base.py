"""The shared base of the torchvision-backed dataset resources.

A dataset resource wraps one split of a torchvision dataset and applies a
transform to each image. Every sample is an ``(image, label)`` pair, so the
data manager's default collate turns a batch into ``[images, labels]`` and
``load_batch`` names them with ``fields: [images, labels]``.

Construction reads (and, with ``download: true``, fetches) the files. The
engine constructs every component in its parent process before any worker
starts, so a download happens once there rather than once per rank.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from training_framework.components import Resource
from training_framework.components.importing import import_object

if TYPE_CHECKING:
    from training_framework.session import Session


#: The transforms every dataset provides; any other value is a dotted path.
TRANSFORM_PRESETS = ("train", "eval")

#: Channel statistics a config can name instead of listing: `mean` and
#: `std` both set to one of these names.
NAMED_STATISTICS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

#: What the presets normalize with when `mean` and `std` are not set.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


@dataclass
class TorchvisionDatasetConfig:
    """`root` and `split` locate the data; `transform` is always stated.

    `transform` is a preset -- ``train`` (random augmentation) or ``eval``
    (deterministic) -- or the dotted path of a transform instance or
    function, which then replaces the preset entirely, normalization
    included. The preset is never inferred from the split, so an evaluation
    split cannot silently be augmented.

    `mean` and `std` normalize the presets' output: three numbers each, or
    both the same name from `NAMED_STATISTICS`. Set both or neither; unset,
    they are ``DEFAULT_MEAN`` / ``DEFAULT_STD``. After parsing they are
    tuples of three floats.
    """

    root: str
    split: str
    transform: str
    download: bool = False
    image_size: int | None = None
    mean: Any = None
    std: Any = None

    #: The splits the dataset accepts; each dataset's config sets its own.
    SPLITS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        if not isinstance(self.root, str) or not self.root:
            raise ValueError(f"root must be a non-empty path; got {self.root!r}")
        if self.split not in self.SPLITS:
            raise ValueError(
                f"split must be one of {list(self.SPLITS)}; got {self.split!r}"
            )
        if not isinstance(self.transform, str) or (
                self.transform not in TRANSFORM_PRESETS
                and "." not in self.transform
        ):
            raise ValueError(
                f"transform must be one of {list(TRANSFORM_PRESETS)} or the "
                "dotted path of a transform (e.g. "
                f"'my_project.transforms.augment'); got {self.transform!r}"
            )
        if not isinstance(self.download, bool):
            raise ValueError(
                f"download must be a boolean; got {self.download!r}"
            )
        if self.image_size is not None and (
                isinstance(self.image_size, bool)
                or not isinstance(self.image_size, int)
                or self.image_size <= 0
        ):
            raise ValueError(
                "image_size must be a positive integer; got "
                f"{self.image_size!r}"
            )
        if (
                self.image_size is not None
                and self.transform not in TRANSFORM_PRESETS
        ):
            raise ValueError(
                "image_size sizes the train/eval presets; a transform given "
                "by dotted path sets its own size, so drop image_size"
            )
        self._resolve_statistics()

    def _resolve_statistics(self) -> None:
        if self.mean is None and self.std is None:
            self.mean, self.std = DEFAULT_MEAN, DEFAULT_STD
            return
        if self.transform not in TRANSFORM_PRESETS:
            raise ValueError(
                "mean and std normalize the train/eval presets; a transform "
                "given by dotted path normalizes itself, so drop them"
            )
        if self.mean is None or self.std is None:
            raise ValueError(
                "mean and std must be set together; got "
                f"mean={self.mean!r}, std={self.std!r}"
            )
        named = [
            value for value in (self.mean, self.std) if isinstance(value, str)
        ]
        if named:
            if self.mean != self.std or self.mean not in NAMED_STATISTICS:
                raise ValueError(
                    "mean and std must both name the same statistics, one of "
                    f"{sorted(NAMED_STATISTICS)}, or both list three "
                    f"numbers; got mean={self.mean!r}, std={self.std!r}"
                )
            self.mean, self.std = NAMED_STATISTICS[self.mean]
            return
        self.mean = _channel_values(self.mean, "mean")
        self.std = _channel_values(self.std, "std")
        if any(value <= 0 for value in self.std):
            raise ValueError(
                f"std values must be greater than 0; got {list(self.std)}"
            )


def _channel_values(value: Any, key: str) -> tuple[float, float, float]:
    """Return `value` as three finite floats, one per RGB channel."""
    if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
    ):
        raise ValueError(
            f"{key} must be three numbers (one per RGB channel) or one of "
            f"{sorted(NAMED_STATISTICS)}; got {value!r}"
        )
    channels = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in channels):
        # YAML reads `.nan` and `.inf` as floats; either would make every
        # normalized pixel non-finite.
        raise ValueError(
            f"{key} values must be finite numbers; got {list(channels)}"
        )
    return channels


class TorchvisionDataset(Dataset, Resource):
    """One split of a torchvision dataset, as a resource.

    A subclass sets its config (for the splits), builds the torchvision
    dataset in `_build_source`, and may change the default image size and
    the train/eval presets. Normalization comes from the config.
    """

    config_schema: ClassVar[type[TorchvisionDatasetConfig]] = (
        TorchvisionDatasetConfig
    )
    #: The side of the square images the presets produce by default.
    IMAGE_SIZE: ClassVar[int] = 224

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self._source = self._build_source()
        self._transform = self._resolve_transform()

    @abstractmethod
    def _build_source(self) -> Any:
        """Return the untransformed torchvision dataset for the split."""
        raise NotImplementedError

    @property
    def split(self) -> str:
        return self._cfg.split

    @property
    def image_size(self) -> int:
        return self._cfg.image_size or self.IMAGE_SIZE

    @property
    def mean(self) -> tuple[float, float, float]:
        """The per-channel mean the presets subtract."""
        return self._cfg.mean

    @property
    def std(self) -> tuple[float, float, float]:
        """The per-channel standard deviation the presets divide by."""
        return self._cfg.std

    @property
    def labels(self) -> list[str]:
        """The class names, indexed by label."""
        return list(self._source.classes)

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def transform(self) -> Callable[[Any], Any]:
        return self._transform

    def train_transform(self, size: int) -> list[Callable[[Any], Any]]:
        """The random augmentation of the ``train`` preset, before
        normalization."""
        return [v2.RandomResizedCrop(size), v2.RandomHorizontalFlip()]

    def eval_transform(self, size: int) -> list[Callable[[Any], Any]]:
        """The deterministic resize of the ``eval`` preset, before
        normalization."""
        return [v2.Resize(round(size * 256 / 224)), v2.CenterCrop(size)]

    def _resolve_transform(self) -> Callable[[Any], Any]:
        choice = self._cfg.transform
        if choice in TRANSFORM_PRESETS:
            build = (
                self.train_transform if choice == "train"
                else self.eval_transform
            )
            return v2.Compose([
                v2.ToImage(),
                *build(self.image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self.mean, self.std),
            ])
        context = f"{self._component_name()}.transform"
        transform = import_object(
            choice, context, example="my_project.transforms.augment",
        )
        if isinstance(transform, type) or not callable(transform):
            raise TypeError(
                f"{context} {choice!r} must name a transform instance or a "
                "function taking an image, not "
                + ("a class" if isinstance(transform, type)
                   else type(transform).__name__)
            )
        return transform

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image, label = self._source[index]
        return self._transform(image), label

    def setup(self, session: Session) -> None:
        pass

    def teardown(self, session: Session) -> None:
        pass
