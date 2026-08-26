"""Shared helpers for the commit-3d45 replacement tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


COMPONENTS_PACKAGE = "tests.test_components"


def register_test_components() -> ModuleType:
    """Register test-only components after the fixture restores the built-ins.

    The test component module is imported normally on the first call and
    reloaded on later calls. This is also compatible with a fresh spawned
    interpreter, where the module is imported by TrainingSession itself.
    """

    existing = sys.modules.get(COMPONENTS_PACKAGE)
    if existing is None:
        return importlib.import_module(COMPONENTS_PACKAGE)
    return importlib.reload(existing)


def session_config(
    root: Path,
    *,
    max_iterations: int,
    event_path: Path | None = None,
    seed: int = 923,
    include_metrics: bool = False,
) -> dict[str, Any]:
    model_config: dict[str, Any] = {
        "initial_weight": 0.25,
        "learning_rate": 0.08,
        "momentum": 0.9,
    }
    train_config: dict[str, Any] = {"noise_scale": 0.07}
    if event_path is not None:
        train_config["event_path"] = str(event_path)

    config: dict[str, Any] = {
        "base_config": {
            "rng_seed": seed,
            "sessions_dir": str(root),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "it_3d45_model": model_config,
        "it_3d45_train": train_config,
    }
    if include_metrics:
        metrics_config: dict[str, Any] = {"call_every": 1}
        if event_path is not None:
            model_config["event_path"] = str(event_path)
            metrics_config["event_path"] = str(event_path)
        config["it_3d45_metrics"] = metrics_config
    return config


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def iteration_events(path: Path) -> list[dict[str, Any]]:
    return [event for event in read_events(path) if event["event"] == "iteration"]


def make_config(tmp_path, max_iterations=2, seed=123):
    return {
        "base_config": {
            "rng_seed": seed,
            "sessions_dir": str(tmp_path),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
        }
    }
