"""`fetch`: bring a batch's results to the workstation.

For each selected task: its record, config, and matrix values, its run
directory without checkpoints (TensorBoard events, dumped configs, component
outputs), its chosen checkpoint(s), and the Slurm logs of every attempt;
plus the batch's plan, receipt, and task tables. Files keep their paths
relative to the batch directory under `<dest>/<batch-id>/`. The file list
is built in one remote command and copied with one resumable transfer, so
running `fetch` again copies only what is new. Fetched checkpoints are
verified against their manifests (SHA-256 of every component file), and
`fetched.json` records what came from where. Nothing on the cluster changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from training_framework.hpc.errors import HpcError
from training_framework.hpc.services.checkpoints import find_checkpoints, newest
from training_framework.hpc.services.status import TERMINAL_STATES, Row, is_task

STATES = ("done", "finished", "all")
CHECKPOINTS = ("latest", "all", "none")
#: Downloads above this size ask first (unless --yes).
CONFIRM_BYTES = 1 << 30
#: Batch-level files fetched with any selection (globs relative to the batch).
BATCH_FILES = ("plan.json", "receipt.json", "data.tsv", "groups/*/tasks.tsv",
               "groups/*/resume-*.tsv")


@dataclass
class FetchTask:
    task: str
    group: str
    index: int
    state: str
    checkpoints: list[tuple[str, int | None]] = field(default_factory=list)  # relative dirs
    logs: list[str] = field(default_factory=list)                            # relative paths


# Selection ---------------------------------------------------------------------------


def _active(row: Row) -> bool:
    return row.live and row.state not in TERMINAL_STATES


def select(receipt: dict, rows: list[Row], index: dict[str, tuple[str, int]],
           names: list[str] | None = None, state: str = "done"):
    """(tasks to fetch, [(task, why skipped)]). `state`: `done` (finished
    successfully), `finished` (any final state), `all` (also tasks still
    queued or running, whose files may be mid-write)."""
    if state not in STATES:
        raise HpcError(f"--state must be one of {', '.join(STATES)}")
    task_rows = {row.name: row for row in rows if is_task(row, receipt) and row.name in index}
    if names:
        unknown = [name for name in names if name not in task_rows]
        if unknown:
            raise HpcError(f"not tasks of this batch: {', '.join(unknown)}")
        considered = list(dict.fromkeys(names))
    else:
        considered = list(task_rows)
    chosen, skipped = [], []
    for name in considered:
        row = task_rows[name]
        if _active(row) and state != "all":
            skipped.append((name, f"still {row.state} (files may be mid-write; --state all to fetch anyway)"))
        elif state == "done" and row.state != "done" and not names:
            skipped.append((name, f"{row.state} (--state finished includes it)"))
        else:
            chosen.append(FetchTask(name, *index[name], row.state))
    return chosen, skipped


def logs_for(receipt: dict, group: str, index: int) -> list[str]:
    """The Slurm log of the task in every attempt that ran it."""
    logs = []
    for node_id, node in receipt["nodes"].items():
        ref = node.get("job_ref")
        if not ref:
            continue
        if node_id == f"group-{group}" or (node.get("attempt") and node.get("group") == group
                                           and index in node.get("tasks", [])):
            logs.append(f"logs/{group}/{ref}_{index}.out")
    return logs


# The file list -----------------------------------------------------------------------

# Arguments: `T<task>` (the task's files, checkpoints excluded), `D<dir>` (every
# file under a directory), `G<glob>` (files matching a shell glob relative to
# the batch, or one path). Only task and checkpoint directories are walked.
# Lines: `F <relative path> <size>`.
_LIST = r'''
cd "$1" || exit 3
shift
for item in "$@"; do
  value=${item#?}
  case "$item" in
    T*) [ -d "tasks/$value" ] && find "tasks/$value" -type f ! -path '*/checkpoints*/*' -printf 'F\t%p\t%s\n' ;;
    D*) [ -d "$value" ] && find "$value" -type f -printf 'F\t%p\t%s\n' ;;
    G*) for f in $value; do [ -f "$f" ] && printf 'F\t%s\t%s\n' "$f" "$(stat -c %s "$f")"; done ;;
  esac
done
true
'''


def plan_files(executor, batch_dir: str, tasks: list[FetchTask], checkpoints: str = "latest",
               includes: list[str] | None = None) -> dict[str, int]:
    """{relative path: size} of everything to fetch except logs (`add_logs`);
    fills each task's `checkpoints`."""
    if checkpoints not in CHECKPOINTS:
        raise HpcError(f"--checkpoints must be one of {', '.join(CHECKPOINTS)}")
    names = [task.task for task in tasks]
    if checkpoints != "none" and tasks:
        found, _ = find_checkpoints(executor, batch_dir, names)
        prefix = batch_dir.rstrip("/") + "/"
        for task in tasks:
            complete = [(path.removeprefix(prefix), it) for path, it in found.get(task.task, [])
                        if it is not None]
            if checkpoints == "all":
                task.checkpoints = sorted(complete, key=lambda item: (item[1], item[0]))
            elif (latest := newest(complete)) is not None:
                task.checkpoints = [latest]
    items = [f"G{pattern}" for pattern in (*BATCH_FILES, *(includes or ()))]
    for task in tasks:
        items.append(f"T{task.task}")
        items += [f"D{path}" for path, _ in task.checkpoints]
    result = executor.run(["sh", "-c", _LIST, "sh", batch_dir, *items])
    if result.status != 0:
        raise HpcError(f"cannot list files in {batch_dir}: {result.stderr.strip()}")
    files: dict[str, int] = {}
    for line in result.stdout.splitlines():
        kind, _, rest = line.partition("\t")
        path, _, size = rest.rpartition("\t")
        if kind == "F" and path and size.isdigit():
            files[path] = int(size)
    return files


def add_logs(executor, batch_dir: str, receipt: dict, tasks: list[FetchTask],
             files: dict[str, int]) -> None:
    """Add each task's Slurm logs (every attempt) that exist on the cluster."""
    wanted = {}
    for task in tasks:
        task.logs = logs_for(receipt, task.group, task.index)
        for path in task.logs:
            wanted[path] = task
    if not wanted:
        return
    result = executor.run(["sh", "-c", _LIST, "sh", batch_dir, *[f"G{path}" for path in wanted]])
    present = {}
    for line in result.stdout.splitlines():
        kind, _, rest = line.partition("\t")
        path, _, size = rest.rpartition("\t")
        if kind == "F" and size.isdigit():
            present[path] = int(size)
    files.update(present)
    for task in tasks:
        task.logs = [path for path in task.logs if path in present]


def already_here(files: dict[str, int], local_root: str) -> tuple[int, int]:
    """(files, bytes) already present locally with the same size."""
    count = size = 0
    for path, length in files.items():
        try:
            if os.path.getsize(os.path.join(local_root, path)) == length:
                count, size = count + 1, size + length
        except OSError:
            continue
    return count, size


# Verification and the index --------------------------------------------------------------


def verify_checkpoint(directory: str) -> list[str]:
    """Problems with a fetched checkpoint: its manifest, `session.pt`, and
    every component file against the SHA-256 the manifest records."""
    try:
        with open(os.path.join(directory, "manifest.json")) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        return [f"manifest.json unreadable: {error}"]
    problems = []
    if not os.path.isfile(os.path.join(directory, "session.pt")):
        problems.append("session.pt is missing")
    for name, record in (manifest.get("components") or {}).items():
        path = os.path.join(directory, record.get("file", ""))
        if not os.path.isfile(path):
            problems.append(f"component '{name}': {record.get('file')} is missing")
            continue
        expected = record.get("sha256")
        if expected:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                problems.append(f"component '{name}': {record.get('file')} does not match its SHA-256")
    return problems


def write_index(local_root: str, receipt: dict, host: str, tasks: list[FetchTask],
                verified: dict[str, list[str]], transferred) -> str:
    """`fetched.json`: what was fetched, from where; earlier fetches of other
    tasks are kept."""
    path = os.path.join(local_root, "fetched.json")
    try:
        with open(path) as handle:
            index = json.load(handle)
    except (OSError, ValueError):
        index = {}
    entries = index.get("tasks", {})
    now = datetime.now(timezone.utc).isoformat()
    for task in tasks:
        checkpoints = [{"path": rel, "iteration": it,
                        "verified": not verified.get(rel),
                        "problems": verified.get(rel, [])} for rel, it in task.checkpoints]
        entries[task.task] = {"state": task.state, "fetched": now, "checkpoints": checkpoints,
                              "logs": task.logs}
    index.update({
        "batch_id": receipt["batch_id"], "cluster": host, "cluster_batch_dir": receipt["batch_dir"],
        "local_batch_dir": os.path.abspath(local_root), "last_fetch": now,
        "last_transfer": {"files": transferred.files, "fetched": transferred.fetched,
                          "bytes": transferred.bytes},
        "tasks": entries,
    })
    with open(path, "w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
