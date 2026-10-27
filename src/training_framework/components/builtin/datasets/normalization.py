"""`normalize_images`: scale and normalize ``uint8`` images on the device.

A dataset with ``image_dtype: uint8`` returns its presets' pixels before
they are scaled and normalized, so batches cross the DataLoader's shared
memory and the host-to-device copy at a quarter of the ``float32`` size.
This step finishes them on the device with the same operations, in the same
order, as the ``float32`` presets, so the images it writes are the ones the
dataset would otherwise have returned.

The mean and std come from the bound dataset, never from this step's own
configuration: one place holds them, so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import torch

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    Step,
    requires_resource,
    step,
)

if TYPE_CHECKING:
    from training_framework.session import Session


@dataclass
class NormalizeImagesConfig:
    """`input` is the key of the ``uint8`` images, `output` the key the
    normalized ``float32`` images are stored under."""

    input: str
    output: str

    def __post_init__(self):
        for name in ("input", "output"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{name} must be a non-empty iteration_context key; got "
                    f"{value!r}"
                )
        if self.input == self.output:
            raise ValueError(
                f"input and output must be different keys; both are "
                f"{self.input!r}"
            )


@requires_resource("dataset")
@step("normalize_images", session_type=TRAINING_SESSION_TYPE)
@step("normalize_images", session_type=ANALYSIS_SESSION_TYPE)
class NormalizeImages(Step):
    """Turn a batch of ``uint8`` images into normalized ``float32`` ones.

    The dataset must return ``uint8`` images (``image_dtype: uint8``);
    anything else is refused when the session is built, since its images
    are either normalized already or not ones this step knows how to scale.
    """

    config_schema = NormalizeImagesConfig

    def __init__(self, config=None):
        super().__init__(config)
        dataset = self.get_dependency("dataset")
        image_dtype = getattr(dataset, "image_dtype", None)
        if image_dtype != "uint8":
            raise ValueError(
                f"{self.name} normalizes uint8 images, but dataset "
                f"'{dataset.name}' returns "
                + (
                    f"{image_dtype} images, which its presets have already "
                    "normalized; set image_dtype: uint8 on the dataset, or "
                    "drop this step"
                    if image_dtype is not None
                    else "images of its own making (no image_dtype); give "
                    "it a train/eval preset with image_dtype: uint8"
                )
            )
        self._mean = tuple(dataset.mean)
        self._std = tuple(dataset.std)
        self._statistics: tuple[torch.Tensor, torch.Tensor] | None = None

    @override
    def context_reads(self) -> dict[str, str]:
        return {self._cfg.input: self._cfg.input}

    @override
    def context_writes(self) -> dict[str, str]:
        return {self._cfg.output: self._cfg.output}

    @override
    def run(self, session: Session, /, *args: Any, **reads: Any) -> Any:
        images = reads[self._cfg.input]
        if not isinstance(images, torch.Tensor) or images.dtype != torch.uint8:
            raise TypeError(
                f"{self.name}.input '{self._cfg.input}' must be a uint8 "
                "tensor of images; got "
                + (
                    f"a {images.dtype} tensor"
                    if isinstance(images, torch.Tensor)
                    else type(images).__name__
                )
            )
        mean, std = self._statistics_on(images.device)
        # torchvision's ToDtype(float32, scale=True) then Normalize, op for
        # op, so the result matches the float32 presets bit for bit.
        return images.to(torch.float32).mul_(1.0 / 255).sub_(mean).div_(std)

    def _statistics_on(
            self, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The mean and std as ``(C, 1, 1)`` tensors on `device`, built once
        per device."""
        if self._statistics is None or self._statistics[0].device != device:
            self._statistics = tuple(
                torch.tensor(values, dtype=torch.float32, device=device)
                .view(-1, 1, 1)
                for values in (self._mean, self._std)
            )
        return self._statistics
