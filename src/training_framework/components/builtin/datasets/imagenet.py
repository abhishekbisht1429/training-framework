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

    Construction lists the class folders only. The images in them (1.28M
    for ``train``) are listed on first use, so each rank lists them once and
    the launching process, which constructs every rank's session, never
    does -- on a network filesystem each listing can take minutes.
    """

    config_schema = ImageNetConfig

    @property
    def _split_root(self) -> str:
        return os.path.join(self._cfg.root, self._cfg.split)

    def _find_labels(self) -> list[str]:
        classes, _ = datasets.folder.find_classes(self._split_root)
        return classes

    def _build_source(self) -> Any:
        source = datasets.ImageFolder(self._split_root)
        if source.classes != self._labels:
            # A label is an index into the classes found at construction; a
            # folder added or removed since would shift every later label.
            raise RuntimeError(
                f"ImageNet class folders in {self._split_root!r} changed "
                f"after the dataset was constructed: found "
                f"{len(source.classes)} classes, expected "
                f"{len(self._labels)}"
            )
        return source
