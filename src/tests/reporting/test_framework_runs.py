"""The report on runs the framework itself wrote: tiny DINO runs launched
by `TrainingEngine` the way a training script launches them, with
`tensorboard` and `tensorboard_summaries`, two learning rates x two seeds,
each in its own local `sessions_dir`."""

from __future__ import annotations

import importlib
import sys

import yaml

from training_framework.engine import Configurator, TrainingEngine
from training_framework.reporting import discover_runs, render_markdown, summarize_runs

SSL_COMPONENTS = "tests.integration.integration_ssl_components"
LR_PATH = "optimizer.optimizer.kwargs.lr"


def _components():
    existing = sys.modules.get(SSL_COMPONENTS)
    if existing is None:
        return importlib.import_module(SSL_COMPONENTS)
    return importlib.reload(existing)


def _train(components, tmp_path, monkeypatch, sessions_dir, *, lr, seed, iterations=3):
    config = components.dino_config(sessions_dir, max_iterations=iterations)
    del config["ddp"]["master_port"]
    config["session_config"]["rng_seed"] = seed
    config["optimizer"]["optimizer"]["kwargs"]["lr"] = lr
    config["tensorboard"] = {"serve": False}
    config["tensorboard_summaries"] = {"scalars": {"train/loss": "loss"}, "optimizer": True}
    config_path = tmp_path / f"{sessions_dir.parent.name}_{sessions_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump({"sessions": [config]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "training-framework", "--config", str(config_path),
        "--heartbeat-timeout", "60", "--process_timeout_on_join", "10",
    ])
    with TrainingEngine(Configurator()) as engine:
        engine.start_session()


def test_local_runs_written_by_the_framework_are_compared_and_grouped(tmp_path, monkeypatch):
    components = _components()
    base_lr = components.dino_config(tmp_path, max_iterations=1)["optimizer"]["optimizer"]["kwargs"]["lr"]
    lrs = (base_lr, base_lr / 2)
    for name, lr in zip(("base", "half"), lrs):
        for seed in (3, 4):
            _train(components, tmp_path, monkeypatch, tmp_path / name / f"seed{seed}", lr=lr, seed=seed)

    summary = summarize_runs(
        discover_runs(tmp_path / "base", tmp_path / "half"),
        ["train/loss:min"], tags=["optimizer/*"],
    )

    assert len(summary.runs) == 4
    assert all(run.run_id.split("/")[0] in ("base", "half") for run in summary.runs)
    assert [(c.name, c.seed) for c in summary.columns] == [
        (LR_PATH, False), ("session_config.rng_seed", True),
    ]
    assert {g.hyperparameters[LR_PATH] for g in summary.groups} == set(lrs)
    for group in summary.groups:
        assert len(group.members) == 2
        assert group.seeds == ["3", "4"]
        assert group.stats[("train/loss", "last")].n == 2
    for run in summary.runs:
        loss = run.metrics["train/loss"]
        assert (loss.points, loss.last_step) == (3, 3)
        assert run.metrics["optimizer/completed_steps"].last == 3.0
    assert summary.problems == []
    assert "## Seed groups" in render_markdown(summary)
