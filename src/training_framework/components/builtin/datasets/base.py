"""The shared base of the torchvision-backed dataset resources.

A dataset resource wraps one split of a torchvision dataset and applies a
transform to each image. Every sample is an ``(image, label)`` pair, so the
data manager's default collate turns a batch into ``[images, labels]`` and
``load_batch`` names them with ``fields: [images, labels]``. With
``transform: views`` the image is a list of augmented views, which collate
into a list of view batches (``fields: [views, labels]``).

By default construction reads (and, with ``download: true``, fetches) the
files. The engine constructs every component in its parent process before
any worker starts, so a download happens there before any rank needs it.
Every rank then constructs its own copy, so construction is repeated, not
shared: a dataset that can name its classes without reading every sample
overrides `_find_labels`, and its samples are then read on first use only --
once per rank, never in the parent.
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

#: The `transform` that returns several augmented views of each image, as
#: the `views` key describes them (self-supervised training).
VIEWS_TRANSFORM = "views"

#: The transforms the dataset builds itself, so `mean`, `std` and
#: `image_dtype` apply to them.
_BUILT_TRANSFORMS = (*TRANSFORM_PRESETS, VIEWS_TRANSFORM)

#: What one group of `views` may set, and the defaults: DINO's and BYOL's
#: shared augmentation (crop, flip, colour jitter, greyscale), plus the blur
#: and solarization each view group chooses.
_VIEW_DEFAULTS: dict[str, Any] = {
    "count": 1,
    "flip_p": 0.5,
    "color_jitter": {
        "p": 0.8, "brightness": 0.4, "contrast": 0.4,
        "saturation": 0.2, "hue": 0.1,
    },
    "grayscale_p": 0.2,
    "blur_p": 0.0,
    "solarize_p": 0.0,
}
_VIEW_REQUIRED = ("size", "scale")
_JITTER_KEYS = ("p", "brightness", "contrast", "saturation", "hue")

#: Channel statistics a config can name instead of listing: `mean` and
#: `std` both set to one of these names.
NAMED_STATISTICS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

#: What the presets normalize with when `mean` and `std` are not set.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)

#: What the presets can return: normalized ``float32`` images, or the
#: ``uint8`` pixels for the `normalize_images` step to normalize on the
#: device -- a quarter of the bytes through shared memory and the copy.
IMAGE_DTYPES = ("float32", "uint8")


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

    `image_dtype` is what the presets return, one of `IMAGE_DTYPES`; unset,
    ``float32``. With ``uint8`` they stop before scaling and normalizing,
    which the `normalize_images` step then does with `mean` and `std`.

    ``transform: views`` returns a list of augmented views of each image
    instead of one, as `views` lists them: groups of ``{count, size, scale,
    flip_p, color_jitter, grayscale_p, blur_p, solarize_p}``, in order.
    `views` is required with it and refused otherwise; each group sets its
    own size, so `image_size` is refused too. `mean`, `std` and
    `image_dtype` apply to every view.
    """

    root: str
    split: str
    transform: str
    download: bool = False
    image_size: int | None = None
    mean: Any = None
    std: Any = None
    image_dtype: str | None = None
    views: Any = None

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
                self.transform not in _BUILT_TRANSFORMS
                and "." not in self.transform
        ):
            raise ValueError(
                f"transform must be one of {list(_BUILT_TRANSFORMS)} or the "
                "dotted path of a transform (e.g. "
                f"'my_project.transforms.augment'); got {self.transform!r}"
            )
        if self.transform == VIEWS_TRANSFORM:
            self.views = _view_groups(self.views)
            if self.image_size is not None:
                raise ValueError(
                    "image_size sizes the train/eval presets; with transform: "
                    "views each group sets its own size, so drop image_size"
                )
        elif self.views is not None:
            raise ValueError(
                "views describes the views of transform: views; set "
                f"transform: views, or drop views (transform is "
                f"{self.transform!r})"
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
        if self.image_dtype is not None:
            if self.image_dtype not in IMAGE_DTYPES:
                raise ValueError(
                    f"image_dtype must be one of {list(IMAGE_DTYPES)}; got "
                    f"{self.image_dtype!r}"
                )
            if self.transform not in _BUILT_TRANSFORMS:
                raise ValueError(
                    "image_dtype sets what the train/eval presets return; a "
                    "transform given by dotted path decides its own output, "
                    "so drop image_dtype"
                )
        self._resolve_statistics()

    def _resolve_statistics(self) -> None:
        if self.mean is None and self.std is None:
            self.mean, self.std = DEFAULT_MEAN, DEFAULT_STD
            return
        if self.transform not in _BUILT_TRANSFORMS:
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


def _view_number(value: Any, name: str, *, low: float, high: float) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
    ):
        raise ValueError(
            f"{name} must be a number between {low} and {high}; got {value!r}"
        )
    return float(value)


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _view_groups(value: Any) -> tuple[dict[str, Any], ...]:
    """Check the `views` groups of ``transform: views``; return them with
    every default filled in."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "transform: views needs views, a non-empty list of view groups "
            "such as {count: 2, size: 224, scale: [0.4, 1.0]}; got "
            f"{value!r}"
        )
    groups = []
    for index, group in enumerate(value):
        name = f"views[{index}]"
        if not isinstance(group, Mapping):
            raise ValueError(f"{name} must be a mapping; got {group!r}")
        allowed = (*_VIEW_REQUIRED, *_VIEW_DEFAULTS)
        unknown = sorted(set(map(str, group)) - set(allowed))
        if unknown:
            raise ValueError(
                f"{name} got unknown keys {unknown}; a view group takes "
                f"{list(allowed)}"
            )
        missing = [key for key in _VIEW_REQUIRED if key not in group]
        if missing:
            raise ValueError(f"{name} needs {missing}")
        scale = group["scale"]
        if not isinstance(scale, (list, tuple)) or len(scale) != 2:
            raise ValueError(
                f"{name}.scale must be [low, high], the fraction of the image "
                f"a crop covers; got {scale!r}"
            )
        low = _view_number(scale[0], f"{name}.scale[0]", low=0.0, high=1.0)
        high = _view_number(scale[1], f"{name}.scale[1]", low=0.0, high=1.0)
        if not 0 < low <= high:
            raise ValueError(
                f"{name}.scale must have 0 < low <= high; got {list(scale)}"
            )
        jitter = group.get("color_jitter", _VIEW_DEFAULTS["color_jitter"])
        if jitter is not None:
            if not isinstance(jitter, Mapping):
                raise ValueError(
                    f"{name}.color_jitter must be a mapping of "
                    f"{list(_JITTER_KEYS)}, or null for none; got {jitter!r}"
                )
            unknown = sorted(set(map(str, jitter)) - set(_JITTER_KEYS))
            if unknown:
                raise ValueError(
                    f"{name}.color_jitter got unknown keys {unknown}; it "
                    f"takes {list(_JITTER_KEYS)}"
                )
            jitter = {**_VIEW_DEFAULTS["color_jitter"], **jitter}
            jitter = {
                key: _view_number(
                    jitter[key], f"{name}.color_jitter.{key}",
                    low=0.0, high=0.5 if key == "hue" else 1.0,
                )
                for key in _JITTER_KEYS
            }
        groups.append({
            "count": _positive_count(group.get("count", 1), f"{name}.count"),
            "size": _positive_count(group["size"], f"{name}.size"),
            "scale": (low, high),
            "color_jitter": jitter,
            **{
                key: _view_number(
                    group.get(key, _VIEW_DEFAULTS[key]), f"{name}.{key}",
                    low=0.0, high=1.0,
                )
                for key in ("flip_p", "grayscale_p", "blur_p", "solarize_p")
            },
        })
    return tuple(groups)


def blur_kernel_size(size: int) -> int:
    """The Gaussian blur kernel for a `size` crop: about a tenth of it, odd,
    at least 3 (BYOL's 23 for 224)."""
    kernel = max(3, round(size / 10))
    return kernel if kernel % 2 else kernel + 1


class ImageViews:
    """Return one augmented view of an image per entry of `transforms`.

    A module-level class rather than a closure, so DataLoader workers
    started by spawn can unpickle the dataset.
    """

    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Any) -> list[Any]:
        return [transform(image) for transform in self.transforms]


class TorchvisionDataset(Dataset, Resource):
    """One split of a torchvision dataset, as a resource.

    A subclass sets its config (for the splits), builds the torchvision
    dataset in `_build_source`, and may change the default image size and
    the train/eval presets. Normalization comes from the config. Overriding
    `_find_labels` defers `_build_source` to the first sample or length
    asked for.
    """

    config_schema: ClassVar[type[TorchvisionDatasetConfig]] = (
        TorchvisionDatasetConfig
    )
    #: The side of the square images the presets produce by default.
    IMAGE_SIZE: ClassVar[int] = 224

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self._source = None
        self._labels = self._find_labels()
        self._transform = self._resolve_transform()

    @abstractmethod
    def _build_source(self) -> Any:
        """Return the untransformed torchvision dataset for the split."""
        raise NotImplementedError

    def _find_labels(self) -> list[str]:
        """Return the class names, indexed by label; run by the constructor.

        The default builds the source to read them, so construction reads
        (or downloads) the files. Override it to name the classes without
        reading every sample -- a model may size its head from `num_classes`
        in its own constructor -- and the source is built on first use.
        """
        return list(self._get_source().classes)

    def _get_source(self) -> Any:
        """The torchvision dataset, built on first use."""
        if self._source is None:
            self._source = self._build_source()
        return self._source

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
    def image_dtype(self) -> str | None:
        """What the presets return, one of `IMAGE_DTYPES`; None for a
        transform given by dotted path, which decides its own output."""
        if self._cfg.transform not in _BUILT_TRANSFORMS:
            return None
        return self._cfg.image_dtype or "float32"

    @property
    def num_views(self) -> int | None:
        """How many views ``transform: views`` returns per image; None for
        any other transform, which returns one image."""
        if self._cfg.transform != VIEWS_TRANSFORM:
            return None
        return sum(view["count"] for view in self._cfg.views)

    @property
    def labels(self) -> list[str]:
        """The class names, indexed by label."""
        return list(self._labels)

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

    def view_transform(self, view: Mapping[str, Any]) -> list[Callable[[Any], Any]]:
        """The augmentation of one view of ``transform: views``, before
        normalization: a random resized crop, a flip, colour jitter,
        greyscale, Gaussian blur and solarization, each with the group's
        probability -- the pipeline DINO and BYOL share."""
        steps: list[Callable[[Any], Any]] = [
            v2.RandomResizedCrop(
                view["size"],
                scale=view["scale"],
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        if view["flip_p"]:
            steps.append(v2.RandomHorizontalFlip(view["flip_p"]))
        jitter = view["color_jitter"]
        if jitter is not None and jitter["p"]:
            steps.append(v2.RandomApply([
                v2.ColorJitter(
                    brightness=jitter["brightness"],
                    contrast=jitter["contrast"],
                    saturation=jitter["saturation"],
                    hue=jitter["hue"],
                ),
            ], p=jitter["p"]))
        if view["grayscale_p"]:
            steps.append(v2.RandomGrayscale(view["grayscale_p"]))
        if view["blur_p"]:
            steps.append(v2.RandomApply([
                v2.GaussianBlur(blur_kernel_size(view["size"]), sigma=(0.1, 2.0)),
            ], p=view["blur_p"]))
        if view["solarize_p"]:
            # On the uint8 pixels: invert every value at or above the middle.
            steps.append(v2.RandomSolarize(128, p=view["solarize_p"]))
        return steps

    def _finish(self, steps: list[Callable[[Any], Any]]) -> v2.Compose:
        """`steps` between conversion to an image and, for float32,
        scaling and normalization."""
        steps = [v2.ToImage(), *steps]
        if self.image_dtype == "float32":
            steps += [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self.mean, self.std),
            ]
        return v2.Compose(steps)

    def _resolve_transform(self) -> Callable[[Any], Any]:
        choice = self._cfg.transform
        if choice in TRANSFORM_PRESETS:
            build = (
                self.train_transform if choice == "train"
                else self.eval_transform
            )
            return self._finish(build(self.image_size))
        if choice == VIEWS_TRANSFORM:
            return ImageViews([
                self._finish(self.view_transform(view))
                for view in self._cfg.views
                for _ in range(view["count"])
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
        return len(self._get_source())

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image, label = self._get_source()[index]
        return self._transform(image), label

    def setup(self, session: Session) -> None:
        pass

    def teardown(self, session: Session) -> None:
        pass
