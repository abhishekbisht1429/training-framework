"""Turning loaded runs into a comparison: the hyperparameters that differ,
each chosen metric per run, and runs that differ only in their seed grouped
into mean ± std."""

from __future__ import annotations

import fnmatch
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from training_framework.reporting.config_paths import (
    canonical,
    covers,
    flatten,
    normalize_path,
)
import re

from training_framework.reporting.config_paths import format_path, parse_path
from training_framework.reporting.runs import Discovery, ReportError, RunRecord, component_entry

DIRECTIONS = ("min", "max")
DEFAULT_SEED_KEYS = ("session_config.rng_seed",)
#: Differ between runs by construction, so they say nothing about a run's
#: hyperparameters: every HPC task has its own `sessions_dir`, and the engine
#: picks a free rendezvous port (and fills in the address) at each launch,
#: wherever the `ddp` entry is listed. `ddp.world_size` is compared: it
#: changes the global batch size.
DEFAULT_IGNORED_KEYS = (
    "session_config.sessions_dir",
    "ddp.master_addr",
    "ddp.master_port",
    "resources.ddp.master_addr",
    "resources.ddp.master_port",
)
#: A cell for a hyperparameter a run's config does not have.
ABSENT = "absent"
#: A cell for a run whose config could not be read.
UNKNOWN = "unknown"
FIELDS = ("last", "best", "mean_last_n")


@dataclass(frozen=True)
class Metric:
    """A tag to report. With a direction, its best value is reported too."""

    tag: str
    direction: str | None = None

    def __post_init__(self):
        if not isinstance(self.tag, str) or not self.tag:
            raise ReportError(f"a metric needs a tag; got {self.tag!r}")
        if self.direction not in (None, *DIRECTIONS):
            raise ReportError(
                f"metric {self.tag!r}: the direction must be min or max; got "
                f"{self.direction!r}"
            )

    @classmethod
    def parse(cls, text: str) -> Metric:
        """`tag`, `tag:min` or `tag:max`."""
        tag, separator, direction = text.rpartition(":")
        if separator and direction in DIRECTIONS:
            return cls(tag, direction)
        return cls(text)


@dataclass(frozen=True)
class Column:
    """A hyperparameter column: a matrix value or a config path."""

    name: str
    source: str          # "matrix" | "config"
    seed: bool


@dataclass
class MetricSummary:
    tag: str
    last: float
    last_step: int
    best: float | None
    best_step: int | None
    mean_last_n: float | None
    points: int
    rewritten: int
    nonfinite: int
    duration_s: float


@dataclass
class Stats:
    """One metric field over a group's members. Non-finite member values are
    left out (`nonfinite` counts them); `missing` counts members without
    the tag."""

    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    n: int
    nonfinite: int
    missing: int


@dataclass
class RunSummary:
    run: RunRecord
    hyperparameters: dict[str, Any]
    metrics: dict[str, MetricSummary]
    group: str | None = None

    @property
    def run_id(self) -> str:
        return self.run.run_id


@dataclass
class GroupSummary:
    name: str
    hyperparameters: dict[str, Any]
    members: list[str]
    seeds: list[str]
    stats: dict[tuple[str, str], Stats]


@dataclass
class Summary:
    roots: list[str]
    metrics: list[Metric]
    last_n: int | None
    columns: list[Column]
    runs: list[RunSummary]
    #: None when runs are not grouped by seed.
    groups: list[GroupSummary] | None
    seed_keys: list[str]
    #: Config paths (and matrix labels) equal in every run with a config.
    shared: dict[str, Any]
    #: Runs whose config could not be read: `shared` says nothing about them.
    unknown_config_runs: list[str]
    ignored_keys: list[str]
    sort: Metric | None
    problems: list[str] = field(default_factory=list)

    @property
    def fields(self) -> list[str]:
        """The metric fields reported, in order."""
        fields = ["last"]
        if any(metric.direction for metric in self.metrics):
            fields.append("best")
        if self.last_n is not None:
            fields.append("mean_last_n")
        return fields


def summarize_runs(
        discovery: Discovery,
        metrics: Sequence[Metric | str] = (),
        *,
        tags: Sequence[str] = (),
        last_n: int | None = None,
        group_by_seed: bool = True,
        seed_keys: Sequence[str] = DEFAULT_SEED_KEYS,
        ignored_keys: Sequence[str] = DEFAULT_IGNORED_KEYS,
        sort: str | None = None,
) -> Summary:
    """Compare the runs `discover_runs` found.

    `metrics` are the tags to report (`Metric` or `"tag[:min|:max]"`);
    `tags` adds every tag matching these globs; with neither, every scalar
    tag is reported. `last_n` adds the mean of each tag's last N points.
    `sort` names a metric with a direction to order runs and groups by.
    """
    runs = discovery.runs
    if not runs:
        raise ReportError(
            "no runs found under " + ", ".join(discovery.roots)
            + (" (" + "; ".join(discovery.problems) + ")" if discovery.problems else "")
        )
    if last_n is not None and (isinstance(last_n, bool) or not isinstance(last_n, int) or last_n <= 0):
        raise ReportError(f"last_n must be a positive integer; got {last_n!r}")
    chosen = _choose_metrics(runs, metrics, tags)
    sort_metric = _sort_metric(sort, chosen)
    ignored = [_path(key, "ignored key") for key in ignored_keys]
    seeds = [_seed_key(key) for key in seed_keys] if group_by_seed else []

    problems = list(discovery.problems)
    flat = {run.run_id: _flat_config(run, ignored) for run in runs}
    # Each run's seeds as config paths: a `component:path` key resolves to
    # where that component's entry is in the run's own config.
    run_seeds = {run.run_id: _config_seeds(run, seeds) for run in runs}
    columns, shared = _columns(runs, flat, seeds, run_seeds)
    if group_by_seed:
        unmatched = [key for key in seeds if not _found_anywhere(key, runs, flat)]
        if unmatched:
            raise ReportError(
                "seed key(s) " + ", ".join(unmatched) + " are in no run's config "
                "or matrix; give the path of the seed with --seed-key, or "
                "turn grouping off with --no-seed-groups"
            )

    summaries = []
    for run in runs:
        for message in run.problems:
            problems.append(f"{run.run_id}: {message}")
        values = {}
        for column in columns:
            values[column.name] = _cell(run, flat[run.run_id], column)
        metric_summaries = {}
        for metric in chosen:
            points = run.scalars.get(metric.tag)
            if not points:
                problems.append(f"{run.run_id}: {metric.tag} is not written by this run")
                continue
            metric_summaries[metric.tag] = _metric_summary(metric, points, last_n, run.rewritten.get(metric.tag, 0))
        summaries.append(RunSummary(run, values, metric_summaries))

    groups = None
    if group_by_seed:
        groups = _groups(summaries, flat, columns, seeds, run_seeds, chosen, last_n, problems)
    if sort_metric is not None:
        summaries.sort(key=lambda summary: _order(
            summary.metrics.get(sort_metric.tag) and summary.metrics[sort_metric.tag].best,
            sort_metric.direction,
        ))
        if groups is not None:
            groups.sort(key=lambda group: _order(
                group.stats[(sort_metric.tag, "best")].mean, sort_metric.direction,
            ))
    if groups is not None:
        # Named in the order shown.
        renamed = {group.name: f"G{index}" for index, group in enumerate(groups, 1)}
        for group in groups:
            group.name = renamed[group.name]
        for summary in summaries:
            summary.group = renamed[summary.group]
        problems = [_rename_groups(message, renamed) for message in problems]
    return Summary(
        roots=list(discovery.roots),
        metrics=chosen,
        last_n=last_n,
        columns=columns,
        runs=summaries,
        groups=groups,
        seed_keys=seeds,
        shared=shared,
        unknown_config_runs=[run.run_id for run in runs if run.config is None],
        ignored_keys=ignored,
        sort=sort_metric,
        problems=problems,
    )


# -- choosing ------------------------------------------------------------------


def _choose_metrics(runs, metrics, tags) -> list[Metric]:
    available = sorted({tag for run in runs for tag in run.scalars})
    chosen: dict[str, Metric] = {}
    for item in metrics:
        metric = Metric.parse(item) if isinstance(item, str) else item
        if not isinstance(metric, Metric):
            raise ReportError(f"a metric is a Metric or 'tag[:min|:max]'; got {item!r}")
        if metric.tag not in available:
            raise ReportError(
                f"no run writes the tag {metric.tag!r}; the tags written are: "
                + (", ".join(available) or "(none)")
            )
        if metric.tag in chosen and chosen[metric.tag] != metric:
            raise ReportError(f"metric {metric.tag!r} is given twice with different directions")
        chosen[metric.tag] = metric
    for pattern in tags:
        if not isinstance(pattern, str) or not pattern:
            raise ReportError(f"a tag glob must be a non-empty string; got {pattern!r}")
        matched = [tag for tag in available if fnmatch.fnmatchcase(tag, pattern)]
        if not matched:
            raise ReportError(
                f"no tag matches {pattern!r}; the tags written are: "
                + (", ".join(available) or "(none)")
            )
        for tag in matched:
            chosen.setdefault(tag, Metric(tag))
    if not metrics and not tags:
        chosen = {tag: Metric(tag) for tag in available}
    if not chosen:
        raise ReportError("the runs found write no scalars")
    return list(chosen.values())


def _sort_metric(sort, chosen) -> Metric | None:
    if sort is None:
        return None
    for metric in chosen:
        if metric.tag == sort:
            if metric.direction is None:
                raise ReportError(
                    f"cannot sort by {sort!r}: give it a direction, "
                    f"--metric {sort}:min or {sort}:max"
                )
            return metric
    raise ReportError(f"cannot sort by {sort!r}: it is not a --metric of this report")


def _path(text, what) -> str:
    try:
        return normalize_path(text)
    except ValueError as error:
        raise ReportError(f"{what}: {error}") from None


# -- hyperparameters -------------------------------------------------------------


def _flat_config(run: RunRecord, ignored: list[str]) -> dict[str, Any] | None:
    if run.config is None:
        return None
    return {
        path: value for path, value in flatten(run.config).items()
        if not any(covers(key, path) for key in ignored)
    }


_LABEL_KEY = re.compile(r'(?P<component>[^.\[\]":]+):(?P<path>.+)\Z')


def _seed_key(text) -> str:
    """A seed key: a config path, or a matrix label `component:path` (the
    path inside that component's entry, wherever the entry is listed)."""
    match = _LABEL_KEY.match(text) if isinstance(text, str) else None
    if match is None:
        return _path(text, "seed key")
    return f"{match.group('component')}:{_path(match.group('path'), 'seed key')}"


def _config_seeds(run: RunRecord, seeds: list[str]) -> list[str]:
    """`seeds` as config paths of this run; a `component:path` key whose
    component the run's config does not have names nothing in it."""
    paths = []
    for key in seeds:
        match = _LABEL_KEY.match(key)
        if match is None:
            paths.append(key)
            continue
        entry = component_entry(run.config, match.group("component"))
        if entry is not None:
            paths.append(format_path(entry + parse_path(match.group("path"))))
    return paths


def _is_seed(name: str, source: str, run_paths: dict[str, str | None], seeds: list[str]) -> bool:
    """Whether a config path, or a matrix label, is a seed: a seed key
    covers the config path; for a matrix value, it is the label itself
    (`data_manager:seed`) or covers where the value is in the run's config
    (`data_manager.seed`, `resources.data_manager.seed`)."""
    if source == "config":
        return any(covers(key, name) for key in seeds)
    path = run_paths.get(name)
    return any(key == name or (path is not None and covers(key, path)) for key in seeds)


def _columns(runs, flat, seeds, run_seeds) -> tuple[list[Column], dict[str, Any]]:
    columns: list[Column] = []
    shared: dict[str, Any] = {}
    # Matrix values first: they are the hyperparameters a batch varies.
    labels: dict[str, None] = {}
    for run in runs:
        labels.update(dict.fromkeys(run.variant))
    tasks = [run for run in runs if run.task is not None]
    for label in labels:
        values = {canonical(run.variant.get(label, ABSENT)) for run in tasks}
        # Where the value is can differ between runs (top level or a group).
        seed = any(
            _is_seed(label, "matrix", run.variant_paths, seeds)
            for run in tasks if label in run.variant
        )
        if len(values) > 1 or len(tasks) < len(runs):
            columns.append(Column(label, "matrix", seed))
        else:
            shared[f"matrix {label}"] = tasks[0].variant[label]
    configs = [config for config in flat.values() if config is not None]
    paths: dict[str, None] = {}
    for config in configs:
        paths.update(dict.fromkeys(config))
    for path in paths:
        values = {canonical(config.get(path, ABSENT)) for config in configs}
        if len(values) > 1:
            seed = any(
                _is_seed(path, "config", {}, run_seeds[run.run_id])
                for run in runs if flat[run.run_id] is not None and path in flat[run.run_id]
            )
            columns.append(Column(path, "config", seed))
        else:
            shared[path] = configs[0][path]
    return columns, shared


def _found_anywhere(key, runs, flat) -> bool:
    """Whether some run has the seed key (a seed equal in every run is a
    shared setting, not a column, and still a seed key that exists)."""
    for run in runs:
        config = flat[run.run_id] or {}
        if any(covers(seed, path) for seed in _config_seeds(run, [key]) for path in config):
            return True
        if any(_is_seed(label, "matrix", run.variant_paths, [key]) for label in run.variant):
            return True
    return False


def _cell(run: RunRecord, config: dict | None, column: Column) -> Any:
    if column.source == "matrix":
        return run.variant.get(column.name, ABSENT)
    if config is None:
        return UNKNOWN
    return config.get(column.name, ABSENT)


# -- metrics ---------------------------------------------------------------------


def _metric_summary(metric, points, last_n, rewritten) -> MetricSummary:
    finite = [point for point in points if math.isfinite(point.value)]
    best = None
    if metric.direction is not None and finite:
        pick = min if metric.direction == "min" else max
        # The earliest step among equal values.
        best = pick(finite, key=lambda point: point.value)
    mean_last_n = None
    if last_n is not None:
        tail = [point.value for point in points[-last_n:]]
        mean_last_n = math.fsum(tail) / len(tail)
    return MetricSummary(
        tag=metric.tag,
        last=points[-1].value,
        last_step=points[-1].step,
        best=None if best is None else best.value,
        best_step=None if best is None else best.step,
        mean_last_n=mean_last_n,
        points=len(points),
        rewritten=rewritten,
        nonfinite=len(points) - len(finite),
        duration_s=points[-1].wall_time - points[0].wall_time,
    )


def _order(value, direction):
    """Sort key: best first, runs without a value last."""
    if value is None or not math.isfinite(value):
        return (1, 0.0)
    return (0, value if direction == "min" else -value)


# -- seed groups -----------------------------------------------------------------


def _groups(summaries, flat, columns, seeds, run_seeds, chosen, last_n, problems) -> list[GroupSummary]:
    keyed: dict[Any, list[RunSummary]] = {}
    for summary in summaries:
        run = summary.run
        config = flat[run.run_id]
        if config is None:
            # Unknown hyperparameters match no other run.
            key = ("unknown config", run.run_id)
        else:
            key = (
                tuple(sorted(
                    (path, canonical(value)) for path, value in config.items()
                    if not any(covers(seed, path) for seed in run_seeds[run.run_id])
                )),
                tuple(sorted(
                    (label, canonical(value)) for label, value in run.variant.items()
                    if not _is_seed(label, "matrix", run.variant_paths, seeds)
                )),
            )
        keyed.setdefault(key, []).append(summary)

    plain_columns = [column for column in columns if not column.seed]
    groups = []
    for index, members in enumerate(keyed.values(), 1):
        name = f"#{index}"
        seed_texts = []
        for member in members:
            member.group = name
            seed_texts.append(_seed_text(member.run, flat[member.run_id], seeds, run_seeds[member.run_id]))
        repeated = sorted({text for text in seed_texts if seed_texts.count(text) > 1})
        for text in repeated:
            runs = [m.run_id for m, s in zip(members, seed_texts) if s == text]
            problems.append(
                f"group {name}: runs {', '.join(runs)} have the same seed ({text}); "
                "a repeated run is counted twice in the group's statistics"
            )
        stats = {}
        for metric in chosen:
            ends = {m.run_id: m.metrics[metric.tag].last_step for m in members if metric.tag in m.metrics}
            if len(set(ends.values())) > 1:
                problems.append(
                    f"group {name}: {metric.tag} ends at different steps ("
                    + ", ".join(f"{run_id}: {step}" for run_id, step in ends.items())
                    + "); its last values are not from the same iteration"
                )
            for name_field in FIELDS:
                if name_field == "best" and metric.direction is None:
                    continue
                if name_field == "mean_last_n" and last_n is None:
                    continue
                stats[(metric.tag, name_field)] = _stats(members, metric.tag, name_field)
        groups.append(GroupSummary(
            name=name,
            hyperparameters={column.name: members[0].hyperparameters[column.name] for column in plain_columns},
            members=[member.run_id for member in members],
            seeds=seed_texts,
            stats=stats,
        ))
    return groups


def _seed_text(run: RunRecord, config: dict | None, seeds: list[str], config_seeds: list[str]) -> str:
    """The run's seed values, `path=value` for each, from its config and its
    matrix values."""
    parts = {}
    for path, value in (config or {}).items():
        if any(covers(seed, path) for seed in config_seeds):
            parts[path] = value
    for label, value in run.variant.items():
        if _is_seed(label, "matrix", run.variant_paths, seeds):
            parts.setdefault(run.variant_paths[label] or label, value)
    if len(parts) == 1:
        return canonical(next(iter(parts.values())))
    return ", ".join(f"{path}={canonical(value)}" for path, value in sorted(parts.items())) or "none"


def _stats(members, tag, name_field) -> Stats:
    values = []
    missing = 0
    for member in members:
        summary = member.metrics.get(tag)
        value = None if summary is None else getattr(summary, name_field)
        if value is None:
            missing += 1
        else:
            values.append(value)
    finite = [value for value in values if math.isfinite(value)]
    return Stats(
        mean=statistics.fmean(finite) if finite else None,
        std=statistics.stdev(finite) if len(finite) > 1 else None,
        min=min(finite) if finite else None,
        max=max(finite) if finite else None,
        n=len(finite),
        nonfinite=len(values) - len(finite),
        missing=missing,
    )


def _rename_groups(message: str, renamed: dict[str, str]) -> str:
    if message.startswith("group #"):
        name, _, rest = message[len("group "):].partition(":")
        return f"group {renamed.get(name, name)}:{rest}"
    return message
