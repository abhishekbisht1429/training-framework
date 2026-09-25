from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from torchvision import datasets
from torchvision.transforms import v2

from training_framework.components import resource
from training_framework.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
)

_NATIVE_SIZE = 32


@dataclass
class CIFAR10Config(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")


@resource("cifar10")
class CIFAR10(TorchvisionDataset):
    """CIFAR-10: 32x32 images of 10 classes (50k train, 10k test).

    The ``train`` preset pads and crops at the native 32 px and flips; both
    presets resize afterwards when `image_size` asks for larger images.
    """

    config_schema = CIFAR10Config
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    IMAGE_SIZE = _NATIVE_SIZE

    def _build_source(self) -> Any:
        return datasets.CIFAR10(
            root=self._cfg.root,
            train=self._cfg.split == "train",
            download=self._cfg.download,
        )

    def train_transform(self, size: int) -> list[Callable[[Any], Any]]:
        return [
            v2.RandomCrop(_NATIVE_SIZE, padding=4),
            v2.RandomHorizontalFlip(),
            *_resized(size),
        ]

    def eval_transform(self, size: int) -> list[Callable[[Any], Any]]:
        return _resized(size)


def _resized(size: int) -> list[Callable[[Any], Any]]:
    return [] if size == _NATIVE_SIZE else [v2.Resize((size, size))]
