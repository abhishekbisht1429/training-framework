# Image Datasets

[← Docs](../README.md) · [Built-in components](builtin-components.md)

Optional resources that fill the `dataset` role with one split of a
torchvision image dataset. Each sample is an `(image, label)` pair with an
`int` class index. With the `train` and `eval` transform presets the image
is a normalized `float32` tensor of shape `(3, image_size, image_size)`; a
transform given by dotted path replaces the whole pipeline, so it decides
the image's type and shape itself.

| Name | Splits | Default `image_size` |
| --- | --- | --- |
| `cifar10` | `train`, `test` | 32 |
| `flowers102` | `train`, `val`, `test` | 224 |
| `stanford_cars` | `train`, `test` | 224 |
| `inaturalist` | `2017`, `2018`, `2019`, `2021_train`, `2021_train_mini`, `2021_valid` | 224 |
| `imagenet` | `train`, `val` | 224 |

Every dataset normalizes with mean and std `[0.5, 0.5, 0.5]` (pixels scaled
to `[-1, 1]`) unless its config sets `mean` and `std`.

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
Naming a dataset without the import fails as an unregistered component, and
the error says which import is missing.

## Configuration

```yaml
role_bindings:
  dataset: cifar10#train

cifar10#train:
  root: /data/cifar10
  split: train
  transform: train       # required: train, eval, views, or a dotted path
  download: true         # optional, default false
  image_size: 224        # optional, default per dataset
  mean: imagenet         # optional, with std: default [0.5, 0.5, 0.5]
  std: imagenet
  image_dtype: float32   # optional: float32 (default) or uint8

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
  - `views`: several augmented views of each image, for self-supervised
    training; see [views](#views-for-self-supervised-training).
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
- `mean`, `std` -- the per-channel normalization of the presets. Set both
  or neither. Each is three numbers (`[0.485, 0.456, 0.406]`), or both are
  the same name: `imagenet` or `cifar10`, for those datasets' published
  statistics. Unset, both are `[0.5, 0.5, 0.5]`. Refused together with a
  dotted-path transform, which does its own normalization.
- `image_dtype` -- what the presets return: `float32` (the default),
  normalized as above, or `uint8`, the pixels before scaling and
  normalizing, for [`normalize_images`](#normalizing-on-the-device) to
  finish on the device. Refused together with a dotted-path transform,
  which decides its own output.

`imagenet` reads `root/<split>/<class folder>/<image>`, with classes and
files in sorted order, so a sample index names the same image on every rank
and machine. Construction lists only the class folders; the images are
listed when the dataset is first read, once per rank, so the launching
process never walks the whole tree. `inaturalist` reads `root/<split>` and labels by species.

Each dataset exposes `labels` (class names, indexed by label),
`num_classes`, `split`, `image_size`, `mean`, `std`, `image_dtype` (`None`
for a dotted-path transform), `num_views` (`None` unless `transform: views`)
and `transform`.

## Normalizing on the device

A `float32` image is four times the bytes of its `uint8` pixels, and every
batch crosses the DataLoader's shared memory (`/dev/shm`) and the copy to
the device at that size. With many workers or large batches that fills
shared memory, and the scaling and normalizing take the workers' CPU time.
`image_dtype: uint8` moves both to the device: the dataset returns the
pixels, and the `normalize_images` step, registered by the same package,
scales and normalizes them there.

```yaml
imagenet#train:
  root: /data/imagenet
  split: train
  transform: train
  mean: imagenet
  std: imagenet
  image_dtype: uint8

data_manager: {batch_size: 128, num_workers: 8, pin_memory: true}
load_batch: {fields: [pixels, labels], non_blocking: true}
normalize_images: {input: pixels, output: images}   # forward reads images
```

- `input`, `output` -- the key of the `uint8` images and the key the
  normalized `float32` images are stored under; required, and different.
- The mean and std are the bound dataset's, never configured here, so the
  two cannot disagree.
- The images it writes are bit for bit those of `image_dtype: float32`: the
  presets crop, resize and flip the `uint8` pixels in both cases, and the
  step scales and normalizes with the same operations in the same order.
  Training is unchanged; only where the work runs moves.
- A dataset that does not return `uint8` images -- `float32`, or a
  dotted-path transform -- is refused when the session is built; a batch
  that is not a `uint8` tensor is refused when the step runs.

## Views for self-supervised training

`transform: views` returns a list of augmented views of each image instead
of one image, as `views` lists them -- what DINO's multi-crop and BYOL's two
views need. The sample is `(views, label)`; the data manager's collate turns
a batch into one batch per view, in order (`load_batch: {fields: [views,
labels]}`), which [`multi_view`](self-supervised.md#multi_view) encodes.

```yaml
imagenet#train:
  root: /data/imagenet
  split: train
  transform: views
  mean: imagenet
  std: imagenet
  views:                                     # groups, in order: global first
    - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 1.0}
    - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 0.1, solarize_p: 0.2}
    - {count: 6, size: 96, scale: [0.05, 0.4], blur_p: 0.5}
```

Each view runs, in this order, each step with its group's probability:

| Key | Default | Step |
|---|---|---|
| `count` | `1` | How many views of this group |
| `size` | required | `RandomResizedCrop` to `size` x `size`, bicubic |
| `scale` | required | `[low, high]`: the fraction of the image a crop covers, `0 < low <= high <= 1` |
| `flip_p` | `0.5` | Horizontal flip |
| `color_jitter` | `{p: 0.8, brightness: 0.4, contrast: 0.4, saturation: 0.2, hue: 0.1}` | Colour jitter, applied with probability `p`; keys left out keep these defaults; `null` for none |
| `grayscale_p` | `0.2` | Greyscale |
| `blur_p` | `0.0` | Gaussian blur, sigma 0.1-2.0, kernel about a tenth of `size` (odd) |
| `solarize_p` | `0.0` | Solarize (invert the pixels at or above 128) |

These are the augmentations DINO and BYOL share; the two methods differ in
the blur and solarization of each view, as in the example above (DINO) and
in [the BYOL example](self-supervised.md#byol). `views` is required with
`transform: views` and refused with any other transform; `image_size` is
refused with it, since each group sets its own size. `mean`, `std` and
`image_dtype` apply to every view: with `uint8`, `normalize_images` takes the
list of view batches and normalizes each.

## Adding a dataset

Subclass `TorchvisionDataset`, give it a config listing its splits, and
build the untransformed torchvision dataset in `_build_source`. Override
`IMAGE_SIZE`, `train_transform` / `eval_transform` (the
steps before normalization), or `labels` where the defaults don't fit.

`_build_source` runs in the constructor by default, since `labels` (and so
`num_classes`, which a model may read in its own constructor) come from the
source. A dataset that can name its classes cheaply -- `imagenet` lists its
class folders -- overrides `_find_labels` to return them, and its source is
then built on the first sample or length asked for:

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
