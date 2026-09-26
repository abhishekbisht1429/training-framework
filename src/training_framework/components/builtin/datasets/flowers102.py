from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from torchvision import datasets

from training_framework.components import resource
from training_framework.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
)


@dataclass
class Flowers102Config(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "val", "test")


@resource("flowers102")
class Flowers102(TorchvisionDataset):
    """Oxford Flowers-102: 102 flower classes (1020 train, 1020 val, 6149
    test). Reading it needs scipy."""

    config_schema = Flowers102Config

    def _build_source(self) -> Any:
        return datasets.Flowers102(
            root=self._cfg.root,
            split=self._cfg.split,
            download=self._cfg.download,
        )
