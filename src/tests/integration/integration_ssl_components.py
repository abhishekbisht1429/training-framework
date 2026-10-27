"""Components and configurations for the DINO and BYOL tests.

A synthetic multi-view dataset (deterministic per sample, so every rank and
every rerun reads the same views), a hook that writes each rank's weights,
and builders for tiny DINO and BYOL configurations on `patch_transformer`.
Used both in-process and as the `components_package` of spawned ranks.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from training_framework.components import (
    Resource,
    SessionHook,
    hook,
    requires_resource,
    resource,
)

EMBED_DIM = 16
OUT_DIM = 32
GLOBAL_SIZE = 32
LOCAL_SIZE = 16
LOSSES = "training_framework.components.builtin.self_supervised.losses"


@resource("issl_views")
class SyntheticViews(Dataset, Resource):
    """`num_samples` images, each returned as `num_global` views of
    `GLOBAL_SIZE` and `num_local` of `LOCAL_SIZE`, and a label."""

    def __init__(self, config=None):
        super().__init__(config)
        config = dict(config or {})
        self.num_samples = config.get("num_samples", 16)
        self.num_global = config.get("num_global", 2)
        self.num_local = config.get("num_local", 0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(1000 + index)
        image = torch.randn(3, GLOBAL_SIZE, GLOBAL_SIZE, generator=generator)
        views = [
            image + 0.1 * torch.randn(image.shape, generator=generator)
            for _ in range(self.num_global)
        ]
        for _ in range(self.num_local):
            top, left = torch.randint(0, GLOBAL_SIZE - LOCAL_SIZE + 1, (2,),
                                      generator=generator).tolist()
            views.append(image[:, top:top + LOCAL_SIZE, left:left + LOCAL_SIZE].clone())
        return views, index % 3

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("ddp")
@requires_resource("model")
@hook("issl_results")
class Results(SessionHook):
    """Write this rank's model parameters (and the DINO centre) at the end."""

    def __init__(self, config=None):
        super().__init__(config)
        self.output_dir = Path(config["output_dir"])

    def pre_session(self, session):
        pass

    def post_session(self, session):
        rank = self.get_dependency("ddp").rank
        model = self.get_dependency("model")
        result: dict[str, Any] = {
            "parameters": {
                name: value.detach().cpu().tolist()
                for name, value in model.state_dict().items()
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"rank_{rank}.json").write_text(
            json.dumps(result), encoding="utf-8",
        )


# -- configurations -----------------------------------------------------------------


def vit(suffix: str, *, dropout: float = 0.0) -> dict[str, Any]:
    """A tiny ViT: 8x8 patches, one layer, a class token."""
    return {
        f"conv_patch_embedding#{suffix}": {
            "in_channels": 3, "patch_size": 8, "embed_dim": EMBED_DIM,
        },
        f"learned_positional_embedding_2d#{suffix}": {
            "grid_size": GLOBAL_SIZE // 8, "embed_dim": EMBED_DIM,
        },
        f"torch_transformer_encoder#{suffix}": {
            "embed_dim": EMBED_DIM, "num_heads": 2, "num_layers": 1,
            "dim_feedforward": 32, "dropout": dropout,
        },
        f"patch_transformer#{suffix}": {
            "class_token": True,
            "dependencies_role_bindings": {
                "patch_embedding": f"conv_patch_embedding#{suffix}",
                "positional_embedding": f"learned_positional_embedding_2d#{suffix}",
                "sequence_encoder": f"torch_transformer_encoder#{suffix}",
            },
        },
    }


def _base(
        sessions_dir: Path,
        *,
        max_iterations: int,
        world_size: int = 1,
        components_package: str = "tests.integration.integration_ssl_components",
        seed: int = 17,
) -> dict[str, Any]:
    return {
        "session_config": {
            "rng_seed": seed,
            "sessions_dir": str(sessions_dir),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": components_package,
            "show_execution_graph": False,
        },
        "ddp": {
            "world_size": world_size, "backend": "gloo",
            "master_addr": "127.0.0.1", "master_port": "12355",
        },
        "data_manager": {
            "batch_size": 4 * world_size, "num_workers": 0, "pin_memory": False,
        },
        "load_batch": {"fields": ["views", "label"]},
        "forward#student": {"args": ["views"], "outputs": "student"},
        "forward#teacher": {
            "method": "teacher_forward", "args": ["views"], "no_grad": True,
            "outputs": "teacher",
        },
        "ema_update": {"momentum": {"start": 0.9, "end": 1.0, "kind": "cosine"}},
    }


def dino_config(
        sessions_dir: Path,
        *,
        max_iterations: int = 3,
        world_size: int = 1,
        num_local: int = 2,
        **overrides: Any,
) -> dict[str, Any]:
    """DINO: multi-crop (2 global + `num_local` local views), prototypes
    head, centring, last layer frozen for the first iteration."""
    config = _base(sessions_dir, max_iterations=max_iterations, world_size=world_size)
    head = {
        "in_dim": EMBED_DIM, "hidden_dim": 32, "out_dim": 8, "num_layers": 3,
        "prototypes": {"num": OUT_DIM},
    }
    config.update({
        "role_bindings": {"model": "ema_teacher", "dataset": "issl_views"},
        "issl_views": {"num_global": 2, "num_local": num_local},
        **vit("student"),
        **vit("teacher"),
        "projection_head#student": copy.deepcopy(head),
        "projection_head#teacher": copy.deepcopy(head),
        **{
            f"backbone_with_head#{suffix}": {
                "pool": "cls",
                "dependencies_role_bindings": {
                    "backbone": f"patch_transformer#{suffix}",
                    "head": f"projection_head#{suffix}",
                },
            }
            for suffix in ("student", "teacher")
        },
        "multi_view#student": {
            "dependencies_role_bindings": {"network": "backbone_with_head#student"},
        },
        "multi_view#teacher": {
            "views": 2,
            "dependencies_role_bindings": {"network": "backbone_with_head#teacher"},
        },
        "ema_teacher": {
            "dependencies_role_bindings": {
                "student": "multi_view#student", "teacher": "multi_view#teacher",
            },
        },
        "dino_distillation_loss": {
            "out_dim": OUT_DIM,
            "num_student_views": 2 + num_local,
            "num_teacher_views": 2,
            "teacher_temp": {"start": 0.04, "end": 0.07, "kind": "linear", "steps": 2},
        },
        "optimizer": {
            "optimizer": {"name": "AdamW", "kwargs": {"lr": 0.01, "weight_decay": 0.04}},
            "param_groups": [
                {"match": ["*.bias", "*norm*"], "kwargs": {"weight_decay": 0.0}},
            ],
        },
        "freeze_gradients": {
            "rules": [{
                "match": "*student.network.head.last_layer.*", "until_iteration": 1,
            }],
        },
        "clip_gradients": {"max_norm": 3.0},
    })
    config.update(overrides)
    return config


def byol_config(
        sessions_dir: Path,
        *,
        max_iterations: int = 3,
        world_size: int = 1,
        **overrides: Any,
) -> dict[str, Any]:
    """BYOL: two views, BatchNorm projector and predictor on the online
    network, a target with the projector only."""
    config = _base(sessions_dir, max_iterations=max_iterations, world_size=world_size)
    projector = {
        "in_dim": EMBED_DIM, "hidden_dim": 32, "out_dim": 8, "num_layers": 2,
        "activation": "relu", "norm": "batch", "init_std": None,
    }
    config.update({
        "role_bindings": {"model": "ema_teacher", "dataset": "issl_views"},
        "issl_views": {"num_global": 2, "num_local": 0},
        **vit("online"),
        **vit("target"),
        "projection_head#online": copy.deepcopy(projector),
        "projection_head#target": copy.deepcopy(projector),
        "projection_head#predictor": {**projector, "in_dim": 8},
        "backbone_with_head#online_proj": {
            "pool": "cls",
            "dependencies_role_bindings": {
                "backbone": "patch_transformer#online",
                "head": "projection_head#online",
            },
        },
        "backbone_with_head#online": {
            "dependencies_role_bindings": {
                "backbone": "backbone_with_head#online_proj",
                "head": "projection_head#predictor",
            },
        },
        "backbone_with_head#target": {
            "pool": "cls",
            "dependencies_role_bindings": {
                "backbone": "patch_transformer#target",
                "head": "projection_head#target",
            },
        },
        "multi_view#online": {
            "dependencies_role_bindings": {"network": "backbone_with_head#online"},
        },
        "multi_view#target": {
            "dependencies_role_bindings": {"network": "backbone_with_head#target"},
        },
        "ema_teacher": {
            "student_part": "network.backbone",
            "teacher_part": "network",
            "dependencies_role_bindings": {
                "student": "multi_view#online", "teacher": "multi_view#target",
            },
        },
        "compute#loss": {
            "function": f"{LOSSES}.byol_loss",
            "args": ["student", "teacher"],
            "constants": {"num_views": 2},
            "outputs": "loss",
        },
        "optimizer": {
            "optimizer": {"name": "AdamW", "kwargs": {"lr": 0.01, "weight_decay": 1e-6}},
        },
    })
    config.update(overrides)
    return config


__all__ = [
    "SyntheticViews",
    "byol_config",
    "dino_config",
    "vit",
]
