"""Fakes for the HPC ports, shared by the contract suites and service tests."""

from __future__ import annotations

import os
import subprocess

from training_framework.hpc.ports import (
    CancelResult,
    Owner,
    SubmissionFailed,
    SubmissionUncertain,
)


class Crash(BaseException):
    """Simulates the submitting process dying (not an ordinary error)."""


class FakeSubmitter:
    """`SchedulerSubmitter` that records calls. `fail_at` (0-based submission
    index) with `failure` in {"definite", "uncertain", "crash"}; `cancel_fails`
    holds job references whose cancellation fails."""

    def __init__(self, fail_at=None, failure="definite", cancel_fails=()):
        self.fail_at = fail_at
        self.failure = failure
        self.cancel_fails = set(cancel_fails)
        self.submitted = []    # (job_ref, node id, script path, dependencies)
        self.cancelled = []

    def submit(self, node, script_path, dependencies):
        index = len(self.submitted)
        if index == self.fail_at:
            if self.failure == "definite":
                raise SubmissionFailed("sbatch: error: invalid partition")
            if self.failure == "uncertain":
                raise SubmissionUncertain("connection lost after sending sbatch")
        job_ref = str(1000 + index)
        self.submitted.append((job_ref, node.id, script_path, list(dependencies)))
        if index == self.fail_at and self.failure == "crash":
            raise Crash()
        return job_ref

    def cancel(self, job_ref):
        self.cancelled.append(job_ref)
        if job_ref in self.cancel_fails:
            return CancelResult(False, f"scancel: error: job {job_ref} not found")
        return CancelResult(True, "cancelled")


class FakeJobProbe:
    """`LeaseOwnerProbe` for job owners: alive while the job is in `running`."""

    def __init__(self, running=()):
        self.running = set(running)

    def is_alive(self, owner: Owner):
        if owner.job_ref is None:
            return None
        return owner.job_ref in self.running


class LocalScheduler:
    """`SchedulerSubmitter` that runs each staged script immediately with
    bash, in dependency order (a task set runs every element in turn)."""

    def __init__(self, extra_env=None):
        self.extra_env = dict(extra_env or {})
        self.results = {}       # node id -> [exit status per element]
        self.outputs = {}       # node id -> [combined output per element]
        self.cancelled = []

    def submit(self, node, script_path, dependencies):
        failed = [dep for dep in dependencies if any(self.results.get(dep, [1]))]
        if failed:
            raise SubmissionFailed(f"dependency failed: {failed}")
        env = {**os.environ, **self.extra_env}
        statuses, outputs = [], []
        elements = range(node.task_count) if node.kind == "task_set" else [None]
        for index in elements:
            if index is not None:
                env["SLURM_ARRAY_TASK_ID"] = str(index)
            result = subprocess.run(["bash", script_path], env=env,
                                    capture_output=True, text=True)
            statuses.append(result.returncode)
            outputs.append(result.stdout + result.stderr)
        self.results[node.id] = statuses
        self.outputs[node.id] = outputs
        return node.id

    def cancel(self, job_ref):
        self.cancelled.append(job_ref)
        return CancelResult(True, "cancelled")
