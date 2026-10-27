"""`multi_view`: run several views of a batch through one network.

Multi-view methods (DINO's multi-crop, BYOL's two views) feed every view of
a batch to the same network. DDP expects one forward pass per backward, so
the views go through in one call of this resource: a list of `(B, ...)`
view batches, global views first. Consecutive views of the same shape are
concatenated and encoded together -- one network call per resolution, as
DINO's `MultiCropWrapper` does -- and the outputs come back concatenated in
view order: `(V * B, ...)`, view `v` at rows `v*B:(v+1)*B`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ModuleResource,
    Resource,
    requires_resource,
    resource,
    role,
)
from training_framework.components.builtin.transformer.modules import (
    _positive_int,
)

role(
    "network",
    Resource,
    description="an nn.Module resource that multi_view runs every view through",
)


@dataclass
class MultiViewConfig:
    """`views` (default: all) is how many leading views to use; a teacher
    that sees only the global crops sets it to their count."""

    views: Any = None

    def __post_init__(self):
        if self.views is not None:
            self.views = _positive_int(self.views, "views")


@requires_resource("network")
@resource("multi_view", session_type=TRAINING_SESSION_TYPE)
@resource("multi_view", session_type=ANALYSIS_SESSION_TYPE)
class MultiView(ModuleResource):
    """Encode a list of view batches with the network bound to `network`.

    The network is attached as the submodule `network` and checkpoints its
    own weights; this component owns none.
    """

    config_schema = MultiViewConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        network = self.get_dependency("network")
        if not isinstance(network, nn.Module):
            raise TypeError(
                f"{self._component_name()} role 'network' is filled by "
                f"{type(network).__name__}, which is not an nn.Module"
            )
        self.network = network

    @property
    def views(self) -> int | None:
        return self._cfg.views

    def forward(self, views: Sequence[torch.Tensor]) -> torch.Tensor:
        name = self._component_name()
        if isinstance(views, torch.Tensor) or not isinstance(views, Sequence):
            raise TypeError(
                f"{name} takes a list of view batches; got a "
                f"{type(views).__name__}"
            )
        if not views or not all(isinstance(view, torch.Tensor) for view in views):
            raise TypeError(f"{name} takes a non-empty list of tensors")
        if self._cfg.views is not None:
            if len(views) < self._cfg.views:
                raise ValueError(
                    f"{name} uses the first {self._cfg.views} views, but got "
                    f"{len(views)}"
                )
            views = views[: self._cfg.views]
        batch = views[0].shape[0]
        if any(view.shape[0] != batch for view in views):
            raise ValueError(
                f"{name}: every view must hold the same batch; got "
                f"{[view.shape[0] for view in views]}"
            )
        outputs = []
        start = 0
        while start < len(views):
            end = start + 1
            while end < len(views) and views[end].shape == views[start].shape:
                end += 1
            output = self.network(torch.cat(views[start:end]))
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{name}: the network returned a {type(output).__name__}, "
                    "not a tensor"
                )
            outputs.append(output)
            start = end
        return torch.cat(outputs)


__all__ = ["MultiView", "MultiViewConfig"]
