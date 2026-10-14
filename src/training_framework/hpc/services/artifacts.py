"""Workstation-side artifact services: each owns its artifact's lifecycle and
recovery policy, using only the outbound ports and the B3 primitive.

- `EnvironmentCache`: site interpreter identity, readiness, setup on the
  login node (the bootstrap builds; a takeover rebuilds).
- `DataUploader`: content-keyed uploads under a lease owned by the
  submitting process (a takeover resumes the transfer).
- `DataPreparer`: readiness, and prepare steps on the login node (a takeover
  re-runs the command).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import socket

from training_framework.hpc.compile.plan import Artifact, ExecutionPlan
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import Owner
from training_framework.hpc.remote import runtime
from training_framework.hpc.remote.lease import (
    KeyMismatch,
    Lease,
    LeaseError,
    check_marker,
    write_marker,
)

IDENTITY_FIELDS = ("implementation", "version", "executable", "platform")
IDENTITY_PROGRAM = (
    "import json, os, platform, sys\n"
    + inspect.getsource(runtime.interpreter_identity)
    + "print(json.dumps(interpreter_identity()))\n"
)


class WorkstationProbe:
    """Owner liveness as seen from the workstation: jobs via the cluster
    probe; processes here directly, elsewhere via the cluster probe (which
    answers only when SSH lands on the owner's host)."""

    def __init__(self, cluster_probe, local_probe):
        self.cluster_probe = cluster_probe
        self.local_probe = local_probe

    def is_alive(self, owner: dict):
        if owner.get("job_ref") is None and owner.get("host") == socket.gethostname():
            return self.local_probe.is_alive(Owner(**owner))
        return self.cluster_probe.is_alive(Owner(**owner))


def probe_identity(executor, context) -> dict:
    result = executor.run([context.interpreter, "-c", IDENTITY_PROGRAM], context)
    if result.status != 0:
        raise HpcError(f"probing {context.interpreter} failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def identity_block(identity: dict) -> str:
    lines = ["env:", "  site_python_identity:"]
    lines += [f"    {field}: {json.dumps(identity.get(field))}" for field in IDENTITY_FIELDS]
    return "\n".join(lines)


class EnvironmentCache:
    def __init__(self, plan: ExecutionPlan, filesystem, executor):
        self.plan = plan
        self.filesystem = filesystem
        self.executor = executor

    @property
    def managed(self) -> bool:
        return self.plan.environment["managed"]

    def check_identity(self) -> None:
        """Before any cache lookup: a declared site identity must match the
        interpreter the bootstrap will use."""
        declared = self.plan.preflight["interpreter"].get("identity")
        if not self.managed or declared is None:
            return
        # Runs before the batch directory exists, so from the remote home.
        context = dataclasses.replace(self.plan.node("setup").context, working_directory=".")
        found = probe_identity(self.executor, context)
        if any(found.get(field) != declared.get(field) for field in IDENTITY_FIELDS):
            raise HpcError(
                "the cluster's site interpreter does not match the profile's "
                f"site_python_identity:\n  declared: {json.dumps(declared, sort_keys=True)}\n"
                f"  found:    {json.dumps(found, sort_keys=True)}\n"
                "If the site changed its Python, update the profile with:\n"
                + identity_block(found)
            )

    def ready(self) -> bool:
        if not self.managed:
            return True
        return check_marker(self.filesystem, self.plan.environment["location"],
                            self.plan.environment["key"])

    def build_on_login(self) -> None:
        context = self.plan.node("setup").context
        result = self.executor.run(
            [context.interpreter, f"{self.plan.batch_dir}/bootstrap.py",
             "--plan", f"{self.plan.batch_dir}/plan.json"], context)
        if result.status != 0:
            output = (result.stdout + result.stderr).strip()[-3000:]
            raise HpcError(f"environment setup failed:\n{output}")


#: Next to an upload in progress: the key being uploaded. rsync never deletes
#: at the destination, so resuming an interrupted upload with different
#: content would leave files the new content does not have.
PENDING_SUFFIX = ".tf-hpc-pending"


class DataUploader:
    def __init__(self, plan: ExecutionPlan, filesystem, transfer, probe, owner: dict):
        self.plan = plan
        self.filesystem = filesystem
        self.transfer = transfer
        self.probe = probe
        self.owner = owner

    def pending(self) -> list[Artifact]:
        pending = []
        for artifact in self.plan.artifacts:
            if artifact.kind != "upload":
                continue
            try:
                if not check_marker(self.filesystem, artifact.location, artifact.key):
                    pending.append(artifact)
            except KeyMismatch as error:
                raise HpcError(f"data source '{artifact.source}': {error}") from error
        return pending

    @staticmethod
    def total_size(artifacts: list[Artifact]) -> int:
        return sum(artifact.details.get("size") or 0 for artifact in artifacts)

    def upload(self, artifact: Artifact) -> int:
        lease = Lease(self.filesystem, self.probe, artifact.location, self.owner)
        try:
            lease.acquire()
        except LeaseError as error:
            raise HpcError(str(error)) from error
        try:
            if check_marker(self.filesystem, artifact.location, artifact.key):
                return 0
            pending_path = artifact.location + PENDING_SUFFIX
            pending = self.filesystem.read(pending_path)
            if pending is not None:
                started = json.loads(pending.decode())["key"]
                if started != artifact.key:
                    raise HpcError(
                        f"data source '{artifact.source}': {artifact.location} holds an "
                        f"interrupted upload of different content (key {started[:12]}, "
                        f"now {artifact.key[:12]}); resuming would keep files the new "
                        f"content does not have. Delete {artifact.location} and "
                        f"{pending_path} on the cluster, then submit again"
                    )
            else:
                self.filesystem.write(pending_path, json.dumps({"key": artifact.key}).encode())
            sent = self.transfer.sync(artifact.details["local"], artifact.location)
            write_marker(self.filesystem, artifact.location, artifact.key)
            self.filesystem.delete(pending_path)
            return sent
        finally:
            lease.release()


class DataPreparer:
    def __init__(self, plan: ExecutionPlan, filesystem, executor):
        self.plan = plan
        self.filesystem = filesystem
        self.executor = executor

    def _artifact(self, node) -> Artifact:
        return next(a for a in self.plan.artifacts if a.kind == "prepare" and a.key == node.artifact)

    def ready(self, node) -> bool:
        artifact = self._artifact(node)
        try:
            return check_marker(self.filesystem, artifact.location, artifact.key)
        except KeyMismatch as error:
            raise HpcError(f"data source '{artifact.source}': {error}") from error

    def run_on_login(self, node) -> None:
        context = node.context
        result = self.executor.run(
            [context.interpreter, "-m", "training_framework.hpc.task", "--prepare", node.id,
             "--plan", f"{self.plan.batch_dir}/plan.json"], context)
        if result.status != 0:
            record = self.filesystem.read(f"{self.plan.batch_dir}/nodes/{node.id}/record.json")
            detail = json.loads(record)["error"] if record else result.stderr.strip()[-2000:]
            raise HpcError(f"prepare step {node.id} failed: {detail}")
