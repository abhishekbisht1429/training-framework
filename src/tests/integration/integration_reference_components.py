"""Components for the reference-training tests.

A tiny ViT classifier on synthetic images, the dataset it reads, and a hook
recording what each iteration did. The model and data are built from fixed
seeds, so a plain-torch loop can rebuild exactly the same starting point.

The module must stay importable by name: a spawned worker imports it as the
session's `components_package`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training_framework.components import (
    Resource,
    StatefulLifeCycleHook,
    StatefulResource,
    hook,
    reads,
    requires_resource,
    resource,
)

NUM_SAMPLES = 32
NUM_CLASSES = 4
IMAGE_SIZE = 16
DATA_SEED = 1234


def dataset_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """The whole dataset: images and their labels."""
    generator = torch.Generator().manual_seed(DATA_SEED)
    images = torch.randn(
        NUM_SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator,
    )
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,), generator=generator)
    return images, labels


class TinyViT(nn.Module):
    """A small vision transformer: patch embedding, class token, pre-norm
    encoder layers without dropout, and a linear head on the class token.

    Parameters are drawn from a generator seeded with `seed`, so two
    instances built with one seed are identical wherever they are built.
    """

    def __init__(
            self,
            *,
            seed: int = 0,
            out_dim: int = NUM_CLASSES,
            dim: int = 32,
            depth: int = 2,
            heads: int = 4,
            patch: int = 4,
    ):
        super().__init__()
        tokens = (IMAGE_SIZE // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, depth, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.05
                )
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.reset_parameters()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        return self.head(self.norm(self.encoder(tokens)[:, 0]))


@resource("ref_vit")
class ReferenceViT(TinyViT, StatefulResource):
    def __init__(self, config=None):
        config = config or {}
        TinyViT.__init__(
            self,
            seed=int(config.get("seed", 0)),
            out_dim=int(config.get("out_dim", NUM_CLASSES)),
        )

    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def set_state(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state)


@resource("ref_dataset")
class ReferenceDataset(Resource):
    """`(image, label, index)`: the index lets a reference replay each
    batch exactly as the data manager drew it."""

    def __init__(self, config=None):
        self._images, self._labels = dataset_tensors()

    def __len__(self) -> int:
        return NUM_SAMPLES

    def __getitem__(self, index: int):
        return self._images[index], int(self._labels[index]), index

    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass


@hook("ref_recorder")
@requires_resource("optimizer")
@reads("loss", "indices")
class ReferenceRecorder(StatefulLifeCycleHook):
    """Record each iteration's loss, batch indices, learning rate after the
    step, and gradient norm (on iterations that step). With `output_path`
    configured, the records are written there as JSON when the session
    ends -- how a spawned worker hands them back."""

    def __init__(self, config=None):
        config = config or {}
        self.call_every = 1
        self._output_path = config.get("output_path")
        self.records: list[dict[str, Any]] = []

    def pre_session(self, session) -> None:
        pass

    def pre_iteration_callback(self, session) -> None:
        pass

    def post_iteration_callback(self, session, *, loss, indices) -> None:
        optimizer = self.get_dependency("optimizer")
        self.records.append({
            "iteration": session.iteration,
            "loss": float(loss.detach()),
            "indices": [int(index) for index in indices.tolist()],
            "lr": optimizer.current_lrs[0],
            "grad_norm": (
                optimizer.grad_norm if optimizer.is_boundary else None
            ),
        })

    def post_session(self, session) -> None:
        if self._output_path is not None:
            Path(self._output_path).write_text(
                json.dumps(self.records), encoding="utf-8",
            )

    def get_state(self) -> dict[str, Any]:
        return {"records": [dict(record) for record in self.records]}

    def set_state(self, state: dict[str, Any]) -> None:
        self.records = [dict(record) for record in state["records"]]
