"""Finding a task's checkpoints on the cluster (used by `resume` and `fetch`).

A checkpoint directory is complete when its `manifest.json` exists (it is
written last; a directory still being written is named `*.tmp`). Each task's
checkpoints live under `tasks/<task>/sessions/session_*/checkpoints*/`, and
a resumed task continues in the same session directory, so this finds the
checkpoints of every attempt.
"""

from __future__ import annotations

import base64

import yaml

from training_framework.hpc.errors import HpcError

# For each task: every complete checkpoint directory (one with manifest.json;
# directories still being written end in `.tmp`) with its iteration, and the
# task's config. Lines: `K <task> <relative dir> <iteration>`,
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
    printf 'K\t%s\t%s\t%s\n' "$t" "$d" "$it"
  done
  [ -f "tasks/$t/config.yaml" ] && printf 'C\t%s\t%s\n' "$t" "$(base64 < "tasks/$t/config.yaml" | tr -d '\n')"
done
true
'''


def find_checkpoints(executor, batch_dir: str, tasks: list[str]):
    """{task: [(absolute dir, iteration)]} and {task: config}, in one command."""
    checkpoints: dict[str, list[tuple[str, int | None]]] = {task: [] for task in tasks}
    configs: dict[str, dict] = {}
    if not tasks:
        return checkpoints, configs
    result = executor.run(["sh", "-c", _FIND, "sh", batch_dir, *tasks])
    if result.status != 0:
        raise HpcError(f"cannot list checkpoints in {batch_dir}: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "K" and len(parts) == 4 and parts[1] in checkpoints:
            iteration = int(parts[3]) if parts[3].isdigit() else None
            checkpoints[parts[1]].append((f"{batch_dir}/{parts[2]}", iteration))
        elif parts[0] == "C" and len(parts) == 3:
            try:
                configs[parts[1]] = yaml.safe_load(base64.b64decode(parts[2]))
            except (ValueError, yaml.YAMLError):
                continue
    return checkpoints, configs


def newest(found: list[tuple[str, int | None]]) -> tuple[str, int] | None:
    """The checkpoint with the highest iteration (ties: the later name)."""
    usable = [item for item in found if item[1] is not None]
    return max(usable, key=lambda item: (item[1], item[0])) if usable else None
