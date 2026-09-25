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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TorchvisionDatasetConfig:
    """`root` and `split` locate the data; `transform` is always stated.

    `transform` is a preset -- ``train`` (random augmentation) or ``eval``
    (deterministic) -- or the dotted path of a transform instance or
    function, which then replaces the preset entirely, normalization
    included. The preset is never inferred from the split, so an evaluation
    split cannot silently be augmented.
    """

    root: str
    split: str
    transform: str
    download: bool = False
    image_size: int | None = None

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


class TorchvisionDataset(Dataset, Resource):
    """One split of a torchvision dataset, as a resource.

    A subclass sets its config (for the splits), builds the torchvision
    dataset in `_build_source`, and may change the normalization
    statistics, the default image size and the train/eval presets.
    """

    config_schema: ClassVar[type[TorchvisionDatasetConfig]] = (
        TorchvisionDatasetConfig
    )
    MEAN: ClassVar[tuple[float, float, float]] = IMAGENET_MEAN
    STD: ClassVar[tuple[float, float, float]] = IMAGENET_STD
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
                v2.Normalize(self.MEAN, self.STD),
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
