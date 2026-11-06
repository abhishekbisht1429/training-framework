"""Finding a task's checkpoints on the cluster (used by `resume` and `fetch`).

A checkpoint directory is complete when its `manifest.json` exists (it is
written last; a directory still being written is named `*.tmp`). Each task's
checkpoints live under `tasks/<task>/sessions/session_*/checkpoints*/`, and
a resumed task continues in the same session directory, so this finds the
checkpoints of every attempt.
"""

from __future__ import annotations

import base64
from typing import NamedTuple

import yaml

from training_framework.hpc.errors import HpcError


class FoundCheckpoint(NamedTuple):
    """A complete checkpoint on the cluster, as its manifest describes it."""
    path: str
    iteration: int | None
    #: The iterations the run was resolved to; None for a checkpoint written
    #: before manifests recorded it.
    total_iterations: int | None = None


# For each task: every complete checkpoint directory (one with manifest.json;
# directories still being written end in `.tmp`) with its iteration and the
# run's total iterations, and the task's config. Lines:
# `K <task> <relative dir> <iteration> <total iterations>`,
# `C <task> <base64 config.yaml>`.
_FIND = r'''
cd "$1" || exit 3
shift
for t in "$@"; do
  for d in tasks/$t/sessions/session_*/checkpoints*/*/; do
    d=${d%/}
    case "$d" in *.tmp) continue;; esac
    [ -f "$d/manifest.json" ] || continue
    it=$(grep -o '"iteration": *[0-9]*' "$d/manifest.json" | head -1 | grep -o '[0-9]*$')
    total=$(grep -o '"total_iterations": *[0-9]*' "$d/manifest.json" | head -1 | grep -o '[0-9]*$')
    printf 'K\t%s\t%s\t%s\t%s\n' "$t" "$d" "$it" "$total"
  done
  [ -f "tasks/$t/config.yaml" ] && printf 'C\t%s\t%s\n' "$t" "$(base64 < "tasks/$t/config.yaml" | tr -d '\n')"
done
true
'''


def find_checkpoints(executor, batch_dir: str, tasks: list[str]):
    """{task: [FoundCheckpoint]} and {task: config}, in one command."""
    checkpoints: dict[str, list[FoundCheckpoint]] = {task: [] for task in tasks}
    configs: dict[str, dict] = {}
    if not tasks:
        return checkpoints, configs
    result = executor.run(["sh", "-c", _FIND, "sh", batch_dir, *tasks])
    if result.status != 0:
        raise HpcError(f"cannot list checkpoints in {batch_dir}: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "K" and len(parts) == 5 and parts[1] in checkpoints:
            checkpoints[parts[1]].append(FoundCheckpoint(
                f"{batch_dir}/{parts[2]}", _count(parts[3]), _count(parts[4]),
            ))
        elif parts[0] == "C" and len(parts) == 3:
            try:
                configs[parts[1]] = yaml.safe_load(base64.b64decode(parts[2]))
            except (ValueError, yaml.YAMLError):
                continue
    return checkpoints, configs


def _count(text: str) -> int | None:
    return int(text) if text.isdigit() else None


def newest(found: list[tuple]) -> tuple | None:
    """The checkpoint with the highest iteration (ties: the later name), of
    items that start `(path, iteration)`."""
    usable = [item for item in found if item[1] is not None]
    return max(usable, key=lambda item: (item[1], item[0])) if usable else None
