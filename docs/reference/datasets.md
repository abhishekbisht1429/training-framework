# Image Datasets

[← Docs](../README.md) · [Built-in components](builtin-components.md)

Optional resources that fill the `dataset` role with one split of a
torchvision image dataset. Each sample is an `(image, label)` pair: a
normalized `float32` tensor of shape `(3, image_size, image_size)` and an
`int` class index.

| Name | Splits | Default `image_size` | Normalization |
| --- | --- | --- | --- |
| `cifar10` | `train`, `test` | 32 | CIFAR-10 statistics |
| `flowers102` | `train`, `val`, `test` | 224 | ImageNet statistics |
| `stanford_cars` | `train`, `test` | 224 | ImageNet statistics |
| `inaturalist` | `2017`, `2018`, `2019`, `2021_train`, `2021_train_mini`, `2021_valid` | 224 | ImageNet statistics |
| `imagenet` | `train`, `val` | 224 | ImageNet statistics |

## Enabling them

They need torchvision (and scipy, for Flowers-102 and Stanford Cars):

```bash
pip install "training-framework[vision]"
```

They are not imported with the other built-ins. Import the package from any
module of your `session_config.components_package`, so the parent process,
every rank and a restored checkpoint all register them:

```python
# my_project/components/__init__.py
import training_framework.components.builtin.datasets  # noqa: F401
```

Without torchvision that import fails with an `ImportError` naming the extra.

## Configuration

```yaml
component_bindings:
  dataset: cifar10#train

cifar10#train:
  root: /data/cifar10
  split: train
  transform: train       # required: train, eval, or a dotted path
  download: true         # optional, default false
  image_size: 224        # optional, default per dataset

data_manager: {batch_size: 128, num_workers: 4, pin_memory: true}
load_batch: {fields: [images, labels]}
```

Two splits of one dataset are two instances, e.g. `cifar10#train` and
`cifar10#test`; bind each consumer to the one it reads.

- `root`, `split` -- where the data lives and which split to read. An
  unknown split is an error that lists the valid ones.
- `transform` -- always stated, never inferred from the split, so an
  evaluation split cannot be augmented by accident:
  - `train`: random augmentation. `RandomResizedCrop` + horizontal flip;
    for `cifar10`, a 4 px padded `RandomCrop` + flip at 32 px.
  - `eval`: deterministic. Resize to `image_size * 256 / 224` and center
    crop; for `cifar10`, nothing beyond normalization.
  - a dotted path (`my_project.transforms.augment`) to a transform instance
    or a function taking a PIL image. It replaces the preset entirely,
    normalization included. A class is refused: name an instance.
- `download` -- fetch the files if missing. Construction runs in the
  engine's parent before any rank starts, so a download happens once.
  `imagenet` refuses it (the data comes from image-net.org); torchvision can
  no longer download `stanford_cars`.
- `image_size` -- the side of the square images the presets produce. For
  `cifar10`, a size other than 32 resizes after augmenting. Refused together
  with a dotted-path transform, which sets its own size.

`imagenet` reads `root/<split>/<class folder>/<image>`, with classes and
files in sorted order, so a sample index names the same image on every rank
and machine. `inaturalist` reads `root/<split>` and labels by species.

Each dataset exposes `labels` (class names, indexed by label),
`num_classes`, `split`, `image_size` and `transform`.

## Adding a dataset

Subclass `TorchvisionDataset`, give it a config listing its splits, and
build the untransformed torchvision dataset in `_build_source`. Override
`MEAN` / `STD` / `IMAGE_SIZE`, `train_transform` / `eval_transform` (the
steps before normalization), or `labels` where the defaults don't fit:

```python
from dataclasses import dataclass
from typing import ClassVar

from torchvision import datasets

from training_framework.components import resource
from training_framework.components.builtin.datasets import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
)


@dataclass
class Food101Config(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")


@resource("food101")
class Food101(TorchvisionDataset):
    config_schema = Food101Config

    def _build_source(self):
        return datasets.Food101(
            root=self._cfg.root,
            split=self._cfg.split,
            download=self._cfg.download,
        )
```
