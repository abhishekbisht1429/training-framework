"""`status`: one row per step and task of a submitted batch, from its
receipt, the plan tables and records on the cluster, and the live queue."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    name: str
    values: str
    state: str
    detail: str


def _variant_text(variant: list) -> str:
    return " ".join(f"{'.'.join(item['address']['path'])}={json.dumps(item['value'])}"
                    for item in variant)


def _records(executor, batch_dir: str) -> dict[str, dict]:
    """Every task and node record under the batch, in one remote command."""
    script = ('cd "$1" 2>/dev/null || exit 0; for f in tasks/*/*/*/record.json '
              'nodes/*/record.json; do [ -f "$f" ] && { printf "%s\\t" "$f"; '
              'tr -d "\\n" < "$f"; echo; }; done; true')
    result = executor.run(["sh", "-c", script, "sh", batch_dir])
    records = {}
    for line in result.stdout.splitlines():
        path, _, text = line.partition("\t")
        try:
            records[path] = json.loads(text)
        except ValueError:
            continue
    return records


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


def batch_status(receipt: dict, filesystem, executor, job_states) -> list[Row]:
    batch_dir = receipt["batch_dir"]
    nodes = receipt["nodes"]
    refs = [node["job_ref"] for node in nodes.values() if node.get("job_ref")]
    live = job_states(refs) if job_states else {}
    records = _records(executor, batch_dir)
    rows = []
    for node_id, node in nodes.items():
        if node_id.startswith("group-"):
            continue
        record = records.get(f"nodes/{node_id}/record.json")
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
        table = filesystem.read(f"{batch_dir}/groups/{group}/tasks.tsv")
        if table is None:
            rows.append(Row(node_id, "", node["status"], "no task table"))
            continue
        for line in table.decode().splitlines()[1:]:
            index, task, variant, config, _hash = line.split("\t")
            values = _variant_text(json.loads(variant))
            record = records.get(f"{config.rsplit('/', 1)[0]}/record.json")
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


def format_rows(rows: list[Row]) -> str:
    headers = ("task", "values", "state", "detail")
    table = [headers] + [(r.name, r.values, r.state, r.detail) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(3)]
    lines = []
    for row in table:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(3)) + "  " + row[3])
    return "\n".join(line.rstrip() for line in lines)
