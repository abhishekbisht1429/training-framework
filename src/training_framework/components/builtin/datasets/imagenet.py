from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from torchvision import datasets

from training_framework.components import resource
from training_framework.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
)


@dataclass
class ImageNetConfig(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "val")

    def __post_init__(self):
        super().__post_init__()
        if self.download:
            raise ValueError(
                "download is not supported: ImageNet must be obtained from "
                "image-net.org and extracted by hand"
            )


@resource("imagenet")
class ImageNet(TorchvisionDataset):
    """ImageNet as extracted class folders: `root/<split>/<wnid>/<image>`.

    Classes and files are read in sorted order, so a sample index names the
    same image on every rank and every machine -- which the distributed
    sampler relies on. Labels are the folder names (WordNet ids).
    """

    config_schema = ImageNetConfig

    def _build_source(self) -> Any:
        return datasets.ImageFolder(
            os.path.join(self._cfg.root, self._cfg.split)
        )
