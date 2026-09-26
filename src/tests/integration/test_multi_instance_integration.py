"""A real run holding one component twice, across a checkpoint and a resume.

Everything else about instances is checked in-process. This drives the actual
engine: the session is built in the parent, pickled, rebuilt in a spawned
worker, checkpointed and rebuilt again by a second launch. If instance
identity did not survive all of that, two instances would come back as one.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from training_framework.engine import Configurator, TrainingEngine


_BASE_COMPONENTS = "tests.integration.integration_training_components"
_COMPONENTS_PACKAGE = "tests.integration.integration_multi_instance_components"


pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="A real spawned run requires PyTorch Gloo support",
)


def _register_integration_components() -> None:
    # Both, in order: `reset_registries` clears the base module's
    # registrations too, and importing this module will not re-run them while
    # it is already in sys.modules.
    for name in (_BASE_COMPONENTS, _COMPONENTS_PACKAGE):
        existing = sys.modules.get(name)
        if existing is None:
            importlib.import_module(name)
        else:
            importlib.reload(existing)


def _session_config(tmp_path, output_dir):
    return {
        "session_config": {
            "rng_seed": 17,
            "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show_execution_graph": False,
        },
        "component_bindings": {
            "model": "integration_ddp_model",
            "dataset": "integration_dataset",
        },
        "integration_ddp_model": {"initial_weight": 0.0},
        "integration_dataset": {"dataset_size": 8},
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "master_addr": "127.0.0.1",
        },
        "data_manager": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
        },
        "checkpointer": {"checkpoint_every": 2},
        "integration_data": {},
        "integration_train": {},
        "integration_loss": {},
        "optimizer": {
            "optimizer": {
                "name": "AdamW",
                "kwargs": {"lr": 0.1, "weight_decay": 0.0},
            },
        },
        # The point of the test: one component, configured twice.
        "integration_instance_recorder#alpha": {
            "label": "alpha",
            "output_dir": str(output_dir),
        },
        "integration_instance_recorder#beta": {
            "label": "beta",
            "output_dir": str(output_dir),
        },
    }


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    with TrainingEngine(Configurator()) as engine:
        engine.start_session()


def _only_checkpoint(tmp_path) -> Path:
    checkpoints = sorted(
        (tmp_path / "sessions").glob("session_*/checkpoints/*")
    )
    assert checkpoints, "the first run wrote no checkpoint"
    return checkpoints[-1]


def _recorded(output_dir, label) -> dict:
    return json.loads(
        (output_dir / f"{label}.json").read_text(encoding="utf-8")
    )


def test_two_instances_stay_distinct_across_a_real_run_and_resume(
        tmp_path,
        monkeypatch,
):
    _register_integration_components()
    output_dir = tmp_path / "instance-results"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "sessions": [_session_config(tmp_path, output_dir)],
        }),
        encoding="utf-8",
    )

    _run(
        [
            "training-framework",
            "--config", str(config_path),
            "--heartbeat-timeout", "30",
            "--process_timeout_on_join", "10",
        ],
        monkeypatch,
    )

    # Each instance ran, under its own name and its own configuration.
    alpha = _recorded(output_dir, "alpha")
    beta = _recorded(output_dir, "beta")
    assert alpha["name"] == "integration_instance_recorder#alpha"
    assert beta["name"] == "integration_instance_recorder#beta"
    assert alpha["implementation"] == beta["implementation"] == (
        "integration_instance_recorder"
    )
    assert alpha["iterations"] == beta["iterations"] == [1, 2]

    # Continue the same session; each instance must restore its own state.
    _run(
        [
            "training-framework",
            "--extend-session", str(_only_checkpoint(tmp_path)),
            "--override", "session_config.max_iterations=4",
            "--heartbeat-timeout", "30",
            "--process_timeout_on_join", "10",
        ],
        monkeypatch,
    )

    resumed_alpha = _recorded(output_dir, "alpha")
    resumed_beta = _recorded(output_dir, "beta")
    assert resumed_alpha["iterations"] == [1, 2, 3, 4]
    assert resumed_beta["iterations"] == [1, 2, 3, 4]
    assert resumed_alpha["name"] == "integration_instance_recorder#alpha"


def test_an_extension_override_reaches_one_instance_of_a_real_run(
        tmp_path,
        monkeypatch,
):
    _register_integration_components()
    output_dir = tmp_path / "override-results"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "sessions": [_session_config(tmp_path, output_dir)],
        }),
        encoding="utf-8",
    )

    _run(
        [
            "training-framework",
            "--config", str(config_path),
            "--heartbeat-timeout", "30",
            "--process_timeout_on_join", "10",
        ],
        monkeypatch,
    )

    # Relabel one instance only. The other keeps writing where it did.
    _run(
        [
            "training-framework",
            "--extend-session", str(_only_checkpoint(tmp_path)),
            "--override",
            "session_config.max_iterations=4",
            "integration_instance_recorder#beta.label=beta_renamed",
            "--heartbeat-timeout", "30",
            "--process_timeout_on_join", "10",
        ],
        monkeypatch,
    )

    assert (output_dir / "beta_renamed.json").exists()
    assert _recorded(output_dir, "beta_renamed")["name"] == (
        "integration_instance_recorder#beta"
    )
    # Untouched, so it still reports the iterations of the extended run.
    assert _recorded(output_dir, "alpha")["iterations"] == [1, 2, 3, 4]
