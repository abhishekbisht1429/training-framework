"""`batches`: every batch you submitted, past and present.

The local receipts (one per `submit`, see `receipt.py`) are the list; they
need no connection. `--live` adds what each batch is doing now, with one SSH
connection per cluster: one scheduler query for all your jobs and one remote
command for every batch's records and task tables (`status.fetch_batches`).
`--remote` also lists batches whose receipts are only on the cluster
(submitted from another machine), from `<roots.batches>/*/receipt.json`.
Nothing is written anywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from training_framework.hpc.services.status import (
    Row,
    fetch_batches,
    missing_files,
    receipt_problem,
    format_rows,
    is_task,
    summarize,
    task_rows,
)

_AGE = re.compile(r"\A(\d+)([mhdw])\Z")
_AGE_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


@dataclass
class BatchView:
    receipt: dict
    origin: str                      # "local" (a receipt here) or "cluster" (only there)
    now: str | None = None           # live summary (`summarize`)
    rows: list[Row] | None = None    # live per-task rows (`task_rows`)
    problem: str | None = None       # why live state is missing

    @property
    def batch_id(self) -> str:
        return self.receipt.get("batch_id", "?")

    @property
    def created(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.receipt["created"])
        except (KeyError, TypeError, ValueError):
            return None


def parse_age(text: str) -> timedelta:
    """`30m`, `12h`, `7d`, `2w`."""
    match = _AGE.match(text or "")
    if match is None:
        raise ValueError(f"{text!r} is not an age such as 30m, 12h, 7d or 2w")
    return timedelta(**{_AGE_UNITS[match.group(2)]: int(match.group(1))})


def load_receipts(directory: str) -> list[BatchView]:
    """Every receipt in the local receipts directory, newest first. A file
    that cannot be read is listed as such rather than skipped."""
    views = []
    if not os.path.isdir(directory):
        return views
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name.startswith("."):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path) as handle:
                receipt = json.load(handle)
            problem = receipt_problem(receipt)
            if problem:
                raise ValueError(f"not a valid receipt ({problem})")
        except (OSError, ValueError) as error:
            receipt = {"batch_id": name[:-len(".json")], "status": "unreadable receipt",
                       "error": f"{path}: {error}", "nodes": {}}
        views.append(BatchView(receipt, "local"))
    return sort_views(views)


def sort_views(views: list[BatchView]) -> list[BatchView]:
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(views, key=lambda view: view.created or oldest, reverse=True)


def remote_receipts(executor, batches_root: str) -> list[dict]:
    """The receipts mirrored in each batch directory under `batches_root`."""
    script = ('for f in "$1"/*/receipt.json; do [ -f "$f" ] && { tr -d "\\n" < "$f"; echo; }; '
              'done; true')
    result = executor.run(["sh", "-c", script, "sh", batches_root])
    receipts = []
    for line in result.stdout.splitlines():
        try:
            receipt = json.loads(line)
        except ValueError:
            continue
        if receipt_problem(receipt) is None:
            receipts.append(receipt)
    return receipts


def select(views: list[BatchView], *, since: timedelta | None = None, name: str | None = None,
           active: bool = False, limit: int | None = None,
           now: datetime | None = None) -> list[BatchView]:
    """Filter: submitted within `since`; batch name containing `name`; only
    batches with queued or running tasks (`active`, needs live state);
    the newest `limit`."""
    now = now or datetime.now(timezone.utc)
    chosen = []
    for view in views:
        if since is not None and (view.created is None or now - view.created > since):
            continue
        if name and name not in view.batch_id:
            continue
        if active and not (view.now or "").startswith(("running", "queued")):
            continue
        chosen.append(view)
    return chosen[:limit] if limit else chosen


def add_live_state(views: list[BatchView], connect, fallback=None) -> None:
    """Fill `now` and `rows` of every view, or `problem` when its state cannot
    be known. `connect(ssh_host, scheduler)` returns (executor, function
    returning all of the user's job states). Receipts that do not record
    their cluster (written before receipts did) use `fallback()`, called at
    most once and only if such a receipt is listed; it returns (ssh_host,
    scheduler) or raises with the reason it cannot."""
    resolved: dict = {}

    def fallback_cluster():
        if "value" not in resolved:
            try:
                resolved["value"] = fallback() if fallback else None
                resolved["error"] = None if resolved["value"] else "no profile was given"
            except Exception as error:   # reported on the receipts that needed it
                resolved["value"], resolved["error"] = None, str(error)
        return resolved["value"], resolved["error"]

    by_cluster: dict[tuple[str, str], list[BatchView]] = {}
    for view in views:
        if view.receipt.get("status") == "unreadable receipt":
            view.problem = view.receipt.get("error")
            continue
        if view.receipt.get("ssh_host"):
            cluster = (view.receipt["ssh_host"], view.receipt.get("scheduler") or "slurm")
        else:
            cluster, error = fallback_cluster()
            if cluster is None:
                view.problem = ("this receipt predates receipts recording their cluster, so the "
                                f"profile is needed ({error}); pass --profile")
                continue
        by_cluster.setdefault(cluster, []).append(view)
    for (host, scheduler), members in by_cluster.items():
        try:
            executor, job_states = connect(host, scheduler)
            live = job_states() if job_states else {}
            files = fetch_batches(executor, [view.receipt["batch_dir"] for view in members])
        except Exception as error:  # reported on each affected batch
            for view in members:
                view.problem = f"cannot read {host}: {type(error).__name__}: {error}"
            continue
        for view in members:
            receipt = view.receipt
            try:
                found = files.get(receipt["batch_dir"])
                problem = missing_files(receipt, found)
                if problem:
                    refs = [node["job_ref"] for node in receipt["nodes"].values() if node.get("job_ref")]
                    queued = any(element == ref or element.startswith(f"{ref}_")
                                 for ref in refs for element in live)
                    view.problem = f"{problem} on {host}" + (
                        "; its jobs are still in the queue" if queued else "")
                    continue
                view.rows = task_rows(receipt, found, live)
                view.now = summarize(receipt, view.rows)
            except Exception as error:  # one bad batch does not hide the others
                view.rows, view.now = None, None
                view.problem = f"cannot summarize: {type(error).__name__}: {error}"


# Output ---------------------------------------------------------------------------


def _local_time(view: BatchView) -> str:
    created = view.created
    return created.astimezone().strftime("%Y-%m-%d %H:%M") if created else "?"


def _submit_text(receipt: dict) -> str:
    status = receipt.get("status", "?")
    error = (receipt.get("error") or "").strip().splitlines()
    if status == "submitted" or not error:
        return status
    reason = error[0].removeprefix("not submitted: ")
    return f"{status} ({reason[:40]}{'...' if len(reason) > 40 else ''})"


def format_views(views: list[BatchView], *, live: bool, tasks: bool, origins: bool) -> str:
    if not views:
        return "no batches"
    headers = ["batch", "submitted", "profile", "tasks", "submit"]
    if origins:
        headers.insert(3, "from")
    if live:
        headers.append("now")
    table = [headers]
    for view in views:
        receipt = view.receipt
        tasks_total = receipt.get("task_count")
        if tasks_total is None and view.rows is not None:     # receipts before task_count
            tasks_total = sum(1 for row in view.rows if is_task(row, receipt))
        row = [view.batch_id, _local_time(view), receipt.get("profile") or "?",
               "?" if tasks_total is None else str(tasks_total), _submit_text(receipt)]
        if origins:
            row.insert(3, "this machine" if view.origin == "local"
                       else receipt.get("submitted_from") or "elsewhere")
        if live:
            row.append(view.now if view.now else f"? ({view.problem})" if view.problem else "?")
        table.append(row)
    widths = [max(len(line[i]) for line in table) for i in range(len(headers) - 1)]
    lines = []
    for index, line in enumerate(table):
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(line, widths))
                     + "  " + line[-1])
        if tasks and index > 0:
            view = views[index - 1]
            if view.rows:
                lines.append(format_rows(view.rows, indent="    "))
                lines.append("")
    return "\n".join(line.rstrip() for line in lines).rstrip()


def as_json(views: list[BatchView], *, tasks: bool) -> list[dict]:
    output = []
    for view in views:
        receipt = view.receipt
        entry = {
            "batch_id": view.batch_id,
            "created": receipt.get("created"),
            "profile": receipt.get("profile"),
            "ssh_host": receipt.get("ssh_host"),
            "batch_dir": receipt.get("batch_dir"),
            "task_count": receipt.get("task_count"),
            "status": receipt.get("status"),
            "error": receipt.get("error"),
            "origin": view.origin,
            "submitted_from": receipt.get("submitted_from"),
            "now": view.now,
            "problem": view.problem,
        }
        if tasks:
            entry["tasks"] = [row.__dict__ for row in view.rows or []]
        output.append(entry)
    return output
