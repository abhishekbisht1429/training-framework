"""`projection_head`: the MLP put on a backbone's features.

One component covers the heads self-supervised methods put on an encoder:

- a DINO head: an MLP to a bottleneck, L2-normalized, then a weight-
  normalized linear layer onto `prototypes.num` outputs;
- a BYOL (or SimCLR) projector or predictor: an MLP with BatchNorm and
  ReLU between its layers.

It takes `(B, in_dim)` features -- a pooled backbone output, which
`backbone_with_head` produces with `pool` -- and returns `(B, out_embed_dim)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from training_framework.components import (
    ModuleResource,
    resource,
)
from training_framework.components.builtin.transformer.modules import (
    _ACTIVATIONS,
    _choice,
    _non_negative_float,
    _positive_int,
)

#: What may sit between the hidden layers, before the activation.
HEAD_NORMS = ("none", "batch", "layer")

_PROTOTYPE_KEYS = ("num", "norm_last_layer")


@dataclass
class ProjectionHeadConfig:
    """`num_layers` linear layers: `in_dim` -> `hidden_dim` ... -> `out_dim`,
    with `norm` and `activation` after every one but the last.

    `prototypes` (default none) adds DINO's last layer: the MLP output is
    L2-normalized and goes through a weight-normalized
    `Linear(out_dim, num, bias=False)`. With `norm_last_layer` (default
    true) its gain is fixed at 1.

    `init_std` initializes every linear layer of the MLP with a truncated
    normal of that standard deviation and zero bias, as DINO does; null keeps
    PyTorch's default initialization (BYOL's).
    """

    in_dim: int
    out_dim: int = 256
    hidden_dim: int = 2048
    num_layers: int = 3
    activation: str = "gelu"
    norm: str = "none"
    prototypes: Any = None
    init_std: Any = 0.02

    def __post_init__(self):
        self.in_dim = _positive_int(self.in_dim, "in_dim")
        self.out_dim = _positive_int(self.out_dim, "out_dim")
        self.hidden_dim = _positive_int(self.hidden_dim, "hidden_dim")
        self.num_layers = _positive_int(self.num_layers, "num_layers")
        self.activation = _choice(self.activation, "activation", tuple(_ACTIVATIONS))
        self.norm = _choice(self.norm, "norm", HEAD_NORMS)
        if self.init_std is not None:
            self.init_std = _non_negative_float(self.init_std, "init_std")
        self.prototypes = _prototypes(self.prototypes)


def _prototypes(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            "prototypes must be null or a mapping {num, norm_last_layer}; got "
            f"{value!r}"
        )
    unknown = sorted(set(map(str, value)) - set(_PROTOTYPE_KEYS))
    if unknown:
        raise ValueError(
            f"prototypes got unknown keys {unknown}; it takes "
            f"{list(_PROTOTYPE_KEYS)}"
        )
    if "num" not in value:
        raise ValueError("prototypes needs num, the number of outputs")
    norm_last_layer = value.get("norm_last_layer", True)
    if not isinstance(norm_last_layer, bool):
        raise ValueError(
            "prototypes.norm_last_layer must be true or false; got "
            f"{norm_last_layer!r}"
        )
    return {
        "num": _positive_int(value["num"], "prototypes.num"),
        "norm_last_layer": norm_last_layer,
    }


class WeightNormLinear(nn.Module):
    """`Linear(in_features, out_features, bias=False)` whose weight is
    `gain * direction / ||direction||`, row by row (DINO's prototype layer).

    The arithmetic of `torch.nn.utils.parametrizations.weight_norm` with
    `dim=0`, kept as two plain parameters: a parametrized module refuses to
    be pickled, and a `ModuleResource` pickles as a module.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        initial = nn.Linear(in_features, out_features, bias=False).weight
        self.direction = nn.Parameter(initial.detach().clone())
        self.gain = nn.Parameter(torch.ones(out_features, 1))

    @property
    def weight(self) -> torch.Tensor:
        return self.gain * self.direction / self.direction.norm(dim=1, keepdim=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.linear(features, self.weight)


@resource("projection_head")
class ProjectionHead(ModuleResource):
    """An MLP head, optionally ending in DINO's weight-normalized prototypes.

    The MLP is `mlp`; the prototype layer, when configured, is `last_layer`
    (a `WeightNormLinear`: parameters `last_layer.gain` and
    `last_layer.direction`), which is what a `freeze_gradients` rule names to
    keep it fixed early in training.
    """

    config_schema = ProjectionHeadConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        cfg = self._cfg
        dims = [cfg.in_dim, *[cfg.hidden_dim] * (cfg.num_layers - 1), cfg.out_dim]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            linear = nn.Linear(in_dim, out_dim)
            if cfg.init_std is not None:
                nn.init.trunc_normal_(linear.weight, std=cfg.init_std)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(dims) - 2:
                if cfg.norm == "batch":
                    layers.append(nn.BatchNorm1d(out_dim))
                elif cfg.norm == "layer":
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(_ACTIVATIONS[cfg.activation]())
        self.mlp = nn.Sequential(*layers)
        if cfg.prototypes is not None:
            self.last_layer = WeightNormLinear(cfg.out_dim, cfg.prototypes["num"])
            if cfg.prototypes["norm_last_layer"]:
                self.last_layer.gain.requires_grad_(False)

    @property
    def in_embed_dim(self) -> int:
        return self._cfg.in_dim

    @property
    def out_embed_dim(self) -> int:
        prototypes = self._cfg.prototypes
        return prototypes["num"] if prototypes is not None else self._cfg.out_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self._cfg.in_dim:
            raise ValueError(
                f"{self._component_name()} expects (B, {self._cfg.in_dim}) "
                f"features; got {tuple(features.shape)}. A backbone returning "
                "tokens needs pooling first (backbone_with_head's pool)."
            )
        output = self.mlp(features)
        if self._cfg.prototypes is None:
            return output
        return self.last_layer(functional.normalize(output, dim=-1, p=2))


__all__ = ["HEAD_NORMS", "ProjectionHead", "ProjectionHeadConfig", "WeightNormLinear"]
