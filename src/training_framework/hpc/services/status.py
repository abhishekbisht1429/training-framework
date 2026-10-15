"""The state of submitted batches: one row per step and task, from the
receipt, the plan tables and records on the cluster, and the live queue.

`status BATCH` shows one batch; `batches` lists many, summarizing each with
`summarize` and optionally showing the same rows. Both read the cluster with
`fetch_batches`, one remote command for any number of batches.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    name: str
    values: str
    state: str
    detail: str


#: Live scheduler states in which a job still occupies or waits for resources.
ACTIVE_STATES = frozenset({"pending", "running", "configuring", "completing", "suspended",
                           "requeued", "resizing", "signaling", "stage_out"})
#: Task states counted together in summaries ("failed (exit 1)" -> "failed").
_SUMMARY_ORDER = ("running", "pending", "done", "interrupted", "failed",
                  "ended without a record", "cancelled")


def _variant_text(variant: list) -> str:
    return " ".join(f"{'.'.join(item['address']['path'])}={json.dumps(item['value'])}"
                    for item in variant)


# Reading the cluster -----------------------------------------------------------

# For each batch directory given: every task and node record, and every
# group's task table. Lines: `R <dir> <relative path> <json>` and
# `T <dir> <group> <base64 of tasks.tsv>`, tab-separated.
_FETCH = r'''
for dir in "$@"; do
  cd "$dir" 2>/dev/null || continue
  for f in tasks/*/*/*/record.json nodes/*/record.json; do
    [ -f "$f" ] && { printf 'R\t%s\t%s\t' "$dir" "$f"; tr -d '\n' < "$f"; echo; }
  done
  for t in groups/*/tasks.tsv; do
    [ -f "$t" ] && { g=${t#groups/}; printf 'T\t%s\t%s\t' "$dir" "${g%/tasks.tsv}"; base64 < "$t" | tr -d '\n'; echo; }
  done
done
true
'''


@dataclass
class BatchFiles:
    """What a batch directory holds: records by relative path, task tables by group."""
    records: dict
    tables: dict


def fetch_batches(executor, batch_dirs: list[str]) -> dict[str, BatchFiles]:
    """Records and task tables of every batch directory, in one remote command.
    A directory that does not exist is simply absent from the result."""
    import base64

    found: dict[str, BatchFiles] = {}
    if not batch_dirs:
        return found
    result = executor.run(["sh", "-c", _FETCH, "sh", *batch_dirs])
    for line in result.stdout.splitlines():
        kind, _, rest = line.partition("\t")
        batch_dir, _, rest = rest.partition("\t")
        key, _, payload = rest.partition("\t")
        files = found.setdefault(batch_dir, BatchFiles({}, {}))
        try:
            if kind == "R":
                files.records[key] = json.loads(payload)
            elif kind == "T":
                files.tables[key] = base64.b64decode(payload).decode()
        except ValueError:      # a record being written, or a damaged file
            continue
    return found


# Rows ---------------------------------------------------------------------------


def _record_state(record: dict) -> tuple[str, str]:
    if "exit_status" in record:
        status = record["exit_status"]
        if record.get("interrupted"):
            return "interrupted", "stopped by the scheduler's advance signal"
        if status == 0:
            runs = record.get("run_directories") or []
            return "done", runs[0] if runs else ""
        error = (record.get("error") or "").strip().splitlines()
        return f"failed (exit {status})", error[-1] if error else ""
    state = record.get("status") or "unknown"
    return ("failed" if state == "failed" else state), record.get("error") or record.get("location", "")


_RANGE = re.compile(r"\A(\d+)_\[([^\]]+)\]\Z")


def expand_ranges(live: dict) -> dict:
    """Also key pending array elements shown as one range (`123_[4-9,12%2]`)
    by each element id (`123_4` ...)."""
    expanded = dict(live)
    for element, state in live.items():
        match = _RANGE.match(element)
        if match is None:
            continue
        job, spec = match.group(1), match.group(2).split("%")[0]
        for part in spec.split(","):
            first, _, last = part.partition("-")
            if first.isdigit() and (not last or last.isdigit()):
                for index in range(int(first), int(last or first) + 1):
                    expanded.setdefault(f"{job}_{index}", state)
    return expanded


def task_rows(receipt: dict, files: BatchFiles | None, live: dict) -> list[Row]:
    """One row per step (setup, prepare) and task of a batch. `files` is what
    `fetch_batches` found for it (None: nothing on the cluster); `live` maps
    job and array element ids to scheduler states."""
    files = files or BatchFiles({}, {})
    live = expand_ranges(live)
    nodes = receipt["nodes"]
    rows = []
    for node_id, node in nodes.items():
        if node_id.startswith("group-"):
            continue
        record = files.records.get(f"nodes/{node_id}/record.json")
        if node.get("job_ref") in live:
            state, detail = live[node["job_ref"]], f"job {node['job_ref']}"
        elif record:
            state, detail = _record_state(record)
        else:
            state, detail = node["status"], ""
        rows.append(Row(node_id, "", state, detail))
    for node_id, node in nodes.items():
        if not node_id.startswith("group-"):
            continue
        group = node_id[len("group-"):]
        table = files.tables.get(group)
        if table is None:
            rows.append(Row(node_id, "", node["status"], "no task table"))
            continue
        for line in table.splitlines()[1:]:
            index, task, variant, config, _hash = line.split("\t")
            values = _variant_text(json.loads(variant))
            record = files.records.get(f"{config.rsplit('/', 1)[0]}/record.json")
            element = f"{node.get('job_ref')}_{index}"
            if record:
                state, detail = _record_state(record)
            elif element in live:
                state, detail = live[element], f"job {element}"
            elif node.get("job_ref") in live:
                state, detail = live[node["job_ref"]], f"job {node['job_ref']}"
            elif node["status"] in ("cancelled", "unknown", "refused", "cancel failed"):
                state, detail = node["status"], ""
            elif node["status"] == "submitted":
                state, detail = "ended without a record", f"see logs/{group}/{node['job_ref']}_{index}.out"
            else:
                state, detail = node["status"], ""
            rows.append(Row(task, values, state, detail))
    return rows


def batch_status(receipt: dict, filesystem, executor, job_states) -> list[Row]:
    """The rows of one batch (`status BATCH`). `job_states` queries the
    scheduler for the receipt's job references (None: no live states)."""
    refs = [node["job_ref"] for node in receipt["nodes"].values() if node.get("job_ref")]
    live = job_states(refs) if job_states else {}
    files = fetch_batches(executor, [receipt["batch_dir"]]).get(receipt["batch_dir"])
    return task_rows(receipt, files, live)


# Summaries ----------------------------------------------------------------------


def _short(state: str) -> str:
    return "failed" if state.startswith("failed") else state


def is_task(row: Row, receipt: dict) -> bool:
    return row.name not in receipt["nodes"] or row.name.startswith("group-")


def summarize(receipt: dict, rows: list[Row]) -> str:
    """One phrase for a batch: what its tasks are doing now."""
    status = receipt.get("status")
    if status == "staged":
        return "not submitted"
    if status == "failed" and not any(n.get("job_ref") for n in receipt["nodes"].values()):
        return "submit failed"
    if status == "submitting":
        return "submit interrupted (no final status)"
    counts = Counter(_short(row.state) for row in rows if is_task(row, receipt))
    steps_failed = [row.name for row in rows
                    if not is_task(row, receipt) and _short(row.state) == "failed"]
    active = sum(count for state, count in counts.items() if state in ACTIVE_STATES)
    phase = ("running" if counts.get("running") else "queued") if active else "finished"
    ordered = sorted(counts.items(), key=lambda item: (
        _SUMMARY_ORDER.index(item[0]) if item[0] in _SUMMARY_ORDER else len(_SUMMARY_ORDER),
        item[0]))
    text = f"{phase}: " + ", ".join(f"{count} {state}" for state, count in ordered)
    if steps_failed:
        text += f" ({', '.join(steps_failed)} failed)"
    if status == "needs reconciliation":
        text += " (needs reconciliation)"
    return text


def format_rows(rows: list[Row], indent: str = "") -> str:
    headers = ("task", "values", "state", "detail")
    table = [headers] + [(r.name, r.values, r.state, r.detail) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(3)]
    lines = []
    for row in table:
        lines.append(indent + "  ".join(row[i].ljust(widths[i]) for i in range(3)) + "  " + row[3])
    return "\n".join(line.rstrip() for line in lines)
