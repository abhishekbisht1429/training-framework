"""Summaries of finished runs, to compare hyperparameters.

`discover_runs` finds the runs under one or more directories (local
`sessions_dir`s, HPC batches, `hpc fetch` results) and reads their
TensorBoard scalars and configs; `summarize_runs` compares them (the
hyperparameters that differ, each metric per run, runs that differ only in
their seed grouped into mean ± std); `render_markdown` / `render_csv` /
`render_json` write the result. From the shell:
`python -m training_framework.reporting summarize ROOT [ROOT ...]`.
"""

from training_framework.reporting.render import render_csv, render_json, render_markdown
from training_framework.reporting.runs import Attempt, Discovery, ReportError, RunRecord, discover_runs
from training_framework.reporting.summary import (
    DEFAULT_IGNORED_KEYS,
    DEFAULT_SEED_KEYS,
    Column,
    GroupSummary,
    Metric,
    MetricSummary,
    RunSummary,
    Stats,
    Summary,
    summarize_runs,
)

__all__ = [
    "DEFAULT_IGNORED_KEYS",
    "DEFAULT_SEED_KEYS",
    "Attempt",
    "Column",
    "Discovery",
    "GroupSummary",
    "Metric",
    "MetricSummary",
    "ReportError",
    "RunRecord",
    "RunSummary",
    "Stats",
    "Summary",
    "discover_runs",
    "render_csv",
    "render_json",
    "render_markdown",
    "summarize_runs",
]
