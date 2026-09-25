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
class StanfordCarsConfig(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")


@resource("stanford_cars")
class StanfordCars(TorchvisionDataset):
    """Stanford Cars: 196 car models (8144 train, 8041 test). Reading it
    needs scipy; torchvision can no longer download it, so place the files
    under `root` by hand."""

    config_schema = StanfordCarsConfig

    def _build_source(self) -> Any:
        return datasets.StanfordCars(
            root=self._cfg.root,
            split=self._cfg.split,
            download=self._cfg.download,
        )
