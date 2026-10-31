"""Finding runs under one or more directories and loading each one: its
TensorBoard scalars, its dumped config, and (for an HPC task) its matrix
values.

A run is a session directory, `<sessions_dir>/session_<timestamp>/`, the
directory a session writes `config.yaml` and its TensorBoard writers'
directories into. A resumed run keeps writing into the same one, adding one
event file per attempt to each writer directory. Nothing here needs HPC
files: an HPC task (`.../tasks/<group>/<experiment>/<NNN>/` holding
`variant.json` and `sessions/session_<timestamp>/`) is a run that also has
a task name and matrix values.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import yaml

from training_framework.components.config import COMPONENT_GROUPS
from training_framework.reporting.config_paths import format_path
from training_framework.reporting.events import Point, read_event_file

SESSION_DIR_PREFIX = "session_"
CONFIG_FILE = "config.yaml"
EXTENSION_CONFIG_GLOB = "config_extension_*.yaml"
VARIANT_FILE = "variant.json"
EVENT_FILE_PREFIX = "events.out.tfevents."


class ReportError(Exception):
    """The report cannot be made as asked (a bad root, filter, or option)."""


@dataclass
class Attempt:
    """One event file of a writer: one launch of the run (a fresh start or a
    resume)."""

    path: str            # relative to the session directory
    first_wall_time: float | None
    points: int
    damage: str | None


@dataclass
class RunRecord:
    run_id: str
    session_dir: str
    task: str | None
    config: dict | None
    #: Matrix values of an HPC task, by label (`component:path` or `path`).
    variant: dict[str, Any]
    #: The same labels, mapped to where the value is in this run's
    #: `config.yaml`: the address's path, under its component's entry (top
    #: level or in a component group) when it names one; None when the
    #: component is not in the config.
    variant_paths: dict[str, str | None]
    #: How many times the run was extended (`config_extension_*.yaml`).
    extensions: int
    #: Writer directories, relative to the session directory.
    writers: list[str]
    attempts: list[Attempt]
    #: Every scalar tag, merged over the attempts, sorted by step.
    scalars: dict[str, list[Point]]
    #: Points a later attempt wrote again (the steps after the checkpoint a
    #: resume started from), per tag; the later attempt's value is kept.
    rewritten: dict[str, int]
    problems: list[str] = field(default_factory=list)


@dataclass
class Discovery:
    roots: list[str]
    runs: list[RunRecord]
    #: Problems that belong to no run (event files outside any session
    #: directory).
    problems: list[str]


@dataclass
class _Located:
    session_dir: str
    root: str
    event_files: list[str] = field(default_factory=list)
    task_dir: str | None = None


def discover_runs(
        *roots: str | os.PathLike,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
) -> Discovery:
    """Every run under `roots`, loaded.

    A root may be a directory of session directories (a `sessions_dir`),
    any directory above them (a batch, `hpc-results/<batch-id>`), or one
    session directory. Run ids are paths relative to the root (an HPC
    task's id is its task name); with more than one root they start with
    the root's directory name. `include`/`exclude` are globs on run ids.
    """
    if not roots:
        raise ReportError("give at least one directory to look for runs in")
    resolved = []
    for root in roots:
        path = os.path.abspath(os.fspath(root))
        if not os.path.isdir(path):
            raise ReportError(f"{os.fspath(root)!r} is not a directory")
        resolved.append(path)
    if len(set(resolved)) != len(resolved):
        raise ReportError("the same directory is given twice")

    problems: list[str] = []
    located: dict[str, _Located] = {}
    for root in resolved:
        for session_dir, entry in _locate(root, problems).items():
            if session_dir in located:
                raise ReportError(
                    f"run {session_dir} is under two of the given directories "
                    f"({located[session_dir].root} and {root}); give "
                    "directories that do not contain one another"
                )
            located[session_dir] = entry

    ids, notes = _run_ids(located, len(resolved) > 1)
    run_problems: dict[str, list[str]] = {}
    for message, session_dir in notes:
        run_problems.setdefault(session_dir, []).append(message)
    for pattern in (*include, *exclude):
        if not isinstance(pattern, str) or not pattern:
            raise ReportError(f"--include/--exclude need a glob; got {pattern!r}")

    runs = []
    for session_dir, run_id in sorted(ids.items(), key=lambda item: item[1]):
        if include and not any(fnmatch.fnmatchcase(run_id, p) for p in include):
            continue
        if any(fnmatch.fnmatchcase(run_id, p) for p in exclude):
            continue
        run = _load(run_id, located[session_dir])
        run.problems[:0] = run_problems.get(session_dir, [])
        runs.append(run)
    return Discovery(roots=resolved, runs=runs, problems=problems)


def _locate(root: str, problems: list[str]) -> dict[str, _Located]:
    """Session directories under `root` with their event files."""
    found: dict[str, _Located] = {}
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        if (
                os.path.basename(directory).startswith(SESSION_DIR_PREFIX)
                and CONFIG_FILE in files
        ):
            found.setdefault(directory, _Located(directory, root))
        for name in sorted(files):
            if not name.startswith(EVENT_FILE_PREFIX):
                continue
            path = os.path.join(directory, name)
            session_dir = _session_dir_of(path, root)
            if session_dir is None:
                problems.append(
                    f"{os.path.relpath(path, root)}: an event file in no "
                    f"{SESSION_DIR_PREFIX}* directory; not reported"
                )
                continue
            found.setdefault(session_dir, _Located(session_dir, root)).event_files.append(path)
    for entry in found.values():
        entry.task_dir = _task_dir_of(entry.session_dir)
    return found


def _session_dir_of(event_file: str, root: str) -> str | None:
    """The nearest directory above `event_file`, up to `root`, named like a
    session directory."""
    directory = os.path.dirname(event_file)
    while True:
        if os.path.basename(directory).startswith(SESSION_DIR_PREFIX):
            return directory
        if directory == root:
            return None
        directory = os.path.dirname(directory)


def _task_dir_of(session_dir: str) -> str | None:
    """The HPC task directory (`<task>/sessions/session_*`, with
    `variant.json`) a session directory belongs to, if any."""
    sessions = os.path.dirname(session_dir)
    if os.path.basename(sessions) != "sessions":
        return None
    task_dir = os.path.dirname(sessions)
    return task_dir if os.path.isfile(os.path.join(task_dir, VARIANT_FILE)) else None


def _task_name(task_dir: str, root: str) -> str:
    """`<group>/<experiment>/<NNN>`: the path after the last `tasks`
    directory, else the path relative to the root."""
    parts = task_dir.split(os.sep)
    if "tasks" in parts[:-1]:
        last = len(parts) - 1 - parts[::-1].index("tasks")
        return "/".join(parts[last + 1:])
    relative = os.path.relpath(task_dir, root)
    return os.path.basename(task_dir) if relative.startswith("..") else relative.replace(os.sep, "/")


def _run_ids(
        located: dict[str, _Located],
        prefix_roots: bool,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Run id by session directory, and (problem, session directory) pairs
    found while naming them."""
    by_task: dict[tuple[str, str], list[str]] = {}
    for session_dir, entry in located.items():
        if entry.task_dir is not None:
            by_task.setdefault((entry.root, _task_name(entry.task_dir, entry.root)), []).append(session_dir)

    ids: dict[str, str] = {}
    notes: list[tuple[str, str]] = []
    for session_dir, entry in located.items():
        if entry.task_dir is not None:
            task = _task_name(entry.task_dir, entry.root)
            siblings = by_task[(entry.root, task)]
            run_id = task
            if len(siblings) > 1:
                run_id = f"{task}/{os.path.basename(session_dir)}"
                notes.append((
                    f"task {task} has {len(siblings)} session directories; "
                    "each is reported as its own run",
                    session_dir,
                ))
        elif session_dir == entry.root:
            run_id = os.path.basename(session_dir)
        else:
            run_id = os.path.relpath(session_dir, entry.root).replace(os.sep, "/")
        if prefix_roots:
            run_id = f"{os.path.basename(entry.root)}/{run_id}"
        ids[session_dir] = run_id

    seen: dict[str, str] = {}
    for session_dir, run_id in ids.items():
        if run_id in seen:
            raise ReportError(
                f"two runs would both be called {run_id!r}: {seen[run_id]} and "
                f"{session_dir}; give the directories separately or under "
                "parents with different names"
            )
        seen[run_id] = session_dir
    return ids, notes


def _load(run_id: str, entry: _Located) -> RunRecord:
    session_dir = entry.session_dir
    problems: list[str] = []
    config = _load_config(session_dir, problems)
    variant, variant_paths = ({}, {})
    task = None
    if entry.task_dir is not None:
        task = _task_name(entry.task_dir, entry.root)
        variant, variant_paths = _load_variant(entry.task_dir, config, problems)
    extensions = len(glob.glob(os.path.join(glob.escape(session_dir), EXTENSION_CONFIG_GLOB)))

    by_writer: dict[str, list[str]] = {}
    for path in entry.event_files:
        writer = os.path.relpath(os.path.dirname(path), session_dir).replace(os.sep, "/")
        by_writer.setdefault(writer, []).append(path)
    if not by_writer:
        problems.append("no TensorBoard event files")
    writers = sorted(by_writer)
    attempts: list[Attempt] = []
    scalars: dict[str, list[Point]] = {}
    rewritten: dict[str, int] = {}
    for writer in writers:
        prefix = f"{writer}/" if len(writers) > 1 else ""
        merged, writer_attempts = _merge_writer(by_writer[writer], session_dir, problems)
        attempts.extend(writer_attempts)
        for tag, (points, count) in merged.items():
            scalars[prefix + tag] = points
            if count:
                rewritten[prefix + tag] = count
    for tag, points in scalars.items():
        bad = sum(1 for point in points if not math.isfinite(point.value))
        if bad:
            problems.append(f"{tag}: {bad} non-finite value(s) (nan/inf)")
    return RunRecord(
        run_id=run_id,
        session_dir=session_dir,
        task=task,
        config=config,
        variant=variant,
        variant_paths=variant_paths,
        extensions=extensions,
        writers=writers,
        attempts=attempts,
        scalars=dict(sorted(scalars.items())),
        rewritten=rewritten,
        problems=problems,
    )


def _merge_writer(
        paths: Iterable[str],
        session_dir: str,
        problems: list[str],
) -> tuple[dict[str, tuple[list[Point], int]], list[Attempt]]:
    """A writer's event files merged by step, the later attempt winning."""
    files = []
    attempts = []
    for path in paths:
        relative = os.path.relpath(path, session_dir).replace(os.sep, "/")
        try:
            read = read_event_file(path)
        except OSError as error:
            problems.append(f"{relative}: cannot be read ({error})")
            continue
        if read.damage is not None:
            problems.append(f"{relative}: {read.damage}")
        files.append((read, relative))
        attempts.append(Attempt(
            path=relative,
            first_wall_time=read.first_wall_time,
            points=sum(len(points) for points in read.scalars.values()),
            damage=read.damage,
        ))
    # Attempts in the order they ran; a file with no scalars has no time
    # and changes nothing.
    files.sort(key=lambda item: (item[0].first_wall_time is None, item[0].first_wall_time or 0.0, item[1]))
    attempts.sort(key=lambda a: (a.first_wall_time is None, a.first_wall_time or 0.0, a.path))

    merged: dict[str, dict[int, Point]] = {}
    rewritten: dict[str, int] = {}
    for read, relative in files:
        for tag, points in read.scalars.items():
            by_step = merged.setdefault(tag, {})
            seen_here: set[int] = set()
            for point in points:
                if point.step in seen_here:
                    problems.append(
                        f"{relative}: {tag} is written twice at step {point.step} "
                        "in one attempt; the later value is used"
                    )
                elif point.step in by_step:
                    rewritten[tag] = rewritten.get(tag, 0) + 1
                seen_here.add(point.step)
                by_step[point.step] = point
    return (
        {
            tag: ([by_step[step] for step in sorted(by_step)], rewritten.get(tag, 0))
            for tag, by_step in merged.items()
        },
        attempts,
    )


def _load_config(session_dir: str, problems: list[str]) -> dict | None:
    path = os.path.join(session_dir, CONFIG_FILE)
    if not os.path.isfile(path):
        problems.append(f"no {CONFIG_FILE}; its hyperparameters are unknown")
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        problems.append(f"{CONFIG_FILE} cannot be read ({error}); its hyperparameters are unknown")
        return None
    if not isinstance(config, dict):
        problems.append(f"{CONFIG_FILE} is not a mapping; its hyperparameters are unknown")
        return None
    return config


def component_entry(config: dict | None, component: str) -> tuple[str, ...] | None:
    """Where `component`'s entry is in a session config: at the top level
    or in one of the component groups; None when it is in neither."""
    if config is None:
        return None
    if component in config and component not in COMPONENT_GROUPS:
        return (component,)
    for group in COMPONENT_GROUPS:
        entries = config.get(group)
        if isinstance(entries, dict) and component in entries:
            return (group, component)
    return None


def _load_variant(
        task_dir: str,
        config: dict | None,
        problems: list[str],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    path = os.path.join(task_dir, VARIANT_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            items = json.load(handle)
        values: dict[str, Any] = {}
        paths: dict[str, str | None] = {}
        for item in items:
            address = item["address"]
            segments = address["path"]
            if isinstance(segments, str):
                segments = [segments]
            path_text = format_path(tuple(segments))
            component = address.get("component")
            label = f"{component}:{path_text}" if component else path_text
            values[label] = item["value"]
            if not component:
                paths[label] = path_text
                continue
            entry = component_entry(config, component)
            if entry is None:
                paths[label] = None
                if config is not None:
                    problems.append(
                        f"matrix value {label}: component {component!r} is not in "
                        f"{CONFIG_FILE}; only the label {label!r} can name it as a seed key"
                    )
            else:
                paths[label] = format_path(entry + tuple(segments))
        return values, paths
    except (OSError, ValueError, TypeError, KeyError) as error:
        problems.append(f"{VARIANT_FILE} cannot be read ({error!r}); its matrix values are unknown")
        return {}, {}
