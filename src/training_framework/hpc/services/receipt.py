"""The submission receipt (building block B11).

Acceptance criterion: no submitted job is ever omitted from the receipt. It
is created before the first remote mutation, and every change rewrites it
atomically: locally always, and mirrored into the batch directory once that
exists.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

from training_framework.hpc.compile.plan import ExecutionPlan

SUBMITTING = "submitting"
SUBMITTED = "submitted"
FAILED = "failed"
NEEDS_RECONCILIATION = "needs reconciliation"
STAGED = "staged"
FINAL_STATUSES = (SUBMITTED, FAILED, NEEDS_RECONCILIATION, STAGED)


def default_receipts_dir() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state",
    )
    return os.path.join(base, "training-framework", "submissions")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Receipt:
    def __init__(self, path: str, data: dict):
        self.path = path
        self.data = data
        self._mirror = None   # (filesystem, remote path)

    @classmethod
    def create(cls, directory: str, plan: ExecutionPlan) -> "Receipt":
        path = os.path.join(directory, f"{plan.batch_id}.json")
        if os.path.exists(path):
            raise FileExistsError(f"a receipt for {plan.batch_id} already exists at {path}")
        receipt = cls(path, {
            "execution_protocol_version": plan.protocol_version,
            "batch_id": plan.batch_id,
            "batch_dir": plan.batch_dir,
            "sha": plan.sha,
            "status": SUBMITTING,
            "staged": False,
            "created": _now(),
            "error": None,
            "nodes": {
                node.id: {
                    "status": "pending",
                    "job_name": f"{plan.batch_id}.{node.id}",
                    "job_ref": None,
                    "depends_on": list(node.depends_on),
                    "cancel": None,
                }
                for node in plan.nodes
            },
            "submission_order": [],
        })
        receipt.save()
        return receipt

    @property
    def status(self) -> str:
        return self.data["status"]

    def node(self, node_id: str) -> dict:
        return self.data["nodes"][node_id]

    def mirror_to(self, filesystem, remote_path: str) -> None:
        self._mirror = (filesystem, remote_path)
        self.save()

    def save(self) -> None:
        payload = json.dumps(self.data, indent=2, sort_keys=True) + "\n"
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        if self._mirror is not None:
            filesystem, remote_path = self._mirror
            try:
                filesystem.write(remote_path, payload.encode())
                self.data.pop("mirror_error", None)
            except Exception as error:  # the local receipt stays authoritative
                self.data["mirror_error"] = f"{type(error).__name__}: {error}"

    # Transitions -----------------------------------------------------------

    def mark_staged(self) -> None:
        self.data["staged"] = True
        self.save()

    def node_submitting(self, node_id: str) -> None:
        self.node(node_id)["status"] = "submitting"
        self.save()

    def node_submitted(self, node_id: str, job_ref: str) -> None:
        node = self.node(node_id)
        node["status"], node["job_ref"] = "submitted", job_ref
        self.data["submission_order"].append(node_id)
        self.save()

    def node_unknown(self, node_id: str) -> None:
        self.node(node_id)["status"] = "unknown"
        self.save()

    def node_cancelled(self, node_id: str, ok: bool, detail: str) -> None:
        node = self.node(node_id)
        node["status"] = "cancelled" if ok else "cancel failed"
        node["cancel"] = {"ok": ok, "detail": detail}
        self.save()

    def finish(self, status: str, error: str | None = None) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError(status)
        self.data["status"] = status
        self.data["error"] = error
        self.data["finished"] = _now()
        self.save()

    def submitted_jobs(self) -> list[tuple[str, str]]:
        return [(node_id, self.node(node_id)["job_ref"]) for node_id in self.data["submission_order"]]
