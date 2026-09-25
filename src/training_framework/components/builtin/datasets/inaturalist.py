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
class INaturalistConfig(TorchvisionDatasetConfig):
    #: torchvision's versions; each is one split, read from `root/<split>`.
    SPLITS: ClassVar[tuple[str, ...]] = (
        "2017",
        "2018",
        "2019",
        "2021_train",
        "2021_train_mini",
        "2021_valid",
    )


@resource("inaturalist")
class INaturalist(TorchvisionDataset):
    """iNaturalist, labelled by species (torchvision's ``full`` category)."""

    config_schema = INaturalistConfig

    def _build_source(self) -> Any:
        return datasets.INaturalist(
            root=self._cfg.root,
            version=self._cfg.split,
            target_type="full",
            download=self._cfg.download,
        )

    @property
    def labels(self) -> list[str]:
        return list(self._source.all_categories)
