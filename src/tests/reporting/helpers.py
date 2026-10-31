"""Run directories laid out the way the framework writes them: a
`session_<timestamp>` directory with `config.yaml` and a TensorBoard writer
directory of event files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.tensorboard import SummaryWriter

WRITER = "Tensorboard_tensorboard"
SESSION = "session_20261031_120000"


def config(*, lr: float = 0.1, seed: int = 1, sessions_dir: str = "runs", **extra: Any) -> dict:
    return {
        "session_config": {
            "rng_seed": seed,
            "sessions_dir": sessions_dir,
            "max_iterations": 10,
            "components_package": "my_project.components",
        },
        "optimizer": {"optimizer": {"kind": "sgd", "kwargs": {"lr": lr}}},
        **extra,
    }


def write_run(
        sessions_dir: Path,
        scalars: dict[str, list[tuple[int, float]]],
        run_config: dict | None = None,
        *,
        session: str = SESSION,
        writer: str = WRITER,
        wall_time: float = 1000.0,
) -> Path:
    """A session directory under `sessions_dir` with `run_config` dumped
    and one attempt writing `scalars` ({tag: [(step, value)]})."""
    session_dir = sessions_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)
    if run_config is not None:
        (session_dir / "config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
    write_attempt(session_dir, scalars, writer=writer, wall_time=wall_time)
    return session_dir


def write_attempt(
        session_dir: Path,
        scalars: dict[str, list[tuple[int, float]]],
        *,
        writer: str = WRITER,
        wall_time: float = 1000.0,
) -> Path:
    """One launch of the run: a new event file in the writer directory."""
    before = set((session_dir / writer).glob("events.out.tfevents.*"))
    summary_writer = SummaryWriter(log_dir=str(session_dir / writer))
    for tag, points in scalars.items():
        for index, (step, value) in enumerate(points):
            summary_writer.add_scalar(tag, value, step, walltime=wall_time + index)
    summary_writer.close()
    (written,) = set((session_dir / writer).glob("events.out.tfevents.*")) - before
    return written


def losses(*values: float, start: int = 1) -> dict[str, list[tuple[int, float]]]:
    return {"loss": [(start + index, value) for index, value in enumerate(values)]}
