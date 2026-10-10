"""The submit service: stage a compiled plan, make its artifacts ready, check
it remotely, and submit its job graph, recording everything in the
submission receipt (B11).

Each check runs before the irreversible action that depends on it (B8):
interpreter identity before any environment cache lookup, artifact markers
before uploads or builds, uploads before any job is queued, preflight before
any training task is queued.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from packaging.specifiers import SpecifierSet

from training_framework.hpc.compile.plan import ExecutionPlan
from training_framework.hpc.compile.render import plan_files
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import (
    RemoteExecutor,
    RemoteFilesystem,
    SchedulerSubmitter,
    SubmissionFailed,
    SubmissionUncertain,
)
from training_framework.hpc.remote import preflight
from training_framework.hpc.services import receipt as receipts
from training_framework.hpc.services.artifacts import DataPreparer, DataUploader, EnvironmentCache
from training_framework.hpc.services.receipt import Receipt


class PreflightFailed(HpcError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("preflight failed:\n" + "\n".join(f"- {p}" for p in problems))


@dataclass(frozen=True)
class Summary:
    """What the user confirms before anything large happens."""

    tasks: int
    groups: int
    uploads: list            # (source, local path, size) pending upload
    upload_bytes: int
    upload_limit: int | None
    environment_ready: bool

    @property
    def over_limit(self) -> bool:
        return self.upload_limit is not None and self.upload_bytes > self.upload_limit


def run_preflight(plan: ExecutionPlan, executor: RemoteExecutor) -> dict:
    """Facts from the environment, compared with the plan's expectations."""
    context = next(node for node in plan.nodes if node.kind == "task_set").context
    result = executor.run(
        [context.interpreter, "-m", "training_framework.hpc.task",
         "--preflight", f"{plan.batch_dir}/plan.json"],
        context,
    )
    if result.status != 0:
        raise PreflightFailed([
            "the environment cannot run training_framework.hpc.task "
            f"(exit {result.status}): {result.stderr.strip()[-2000:]}"
        ])
    try:
        facts = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise PreflightFailed([f"preflight printed no facts: {result.stdout[-2000:]}"])
    problems = preflight.compare(facts, plan.preflight, f"{plan.batch_dir}/source")
    requires = plan.preflight.get("requires_python")
    version = (facts.get("interpreter") or {}).get("version")
    if requires and version and not SpecifierSet(requires).contains(version, prereleases=True):
        problems.append(
            f"the environment's Python {version} does not satisfy requires-python '{requires}'"
        )
    if problems:
        raise PreflightFailed(problems)
    return facts


def _cancel_recorded(receipt: Receipt, submitter: SchedulerSubmitter) -> None:
    for node_id, job_ref in reversed(receipt.submitted_jobs()):
        try:
            result = submitter.cancel(job_ref)
            receipt.node_cancelled(node_id, result.ok, result.detail)
        except Exception as error:  # recorded, never swallowed
            receipt.node_cancelled(node_id, False, f"{type(error).__name__}: {error}")


def submit(
        plan: ExecutionPlan,
        manifest_raw: dict,
        *,
        stager,
        filesystem: RemoteFilesystem,
        executor: RemoteExecutor,
        submitter: SchedulerSubmitter,
        transfer=None,
        probe=None,
        owner: dict | None = None,
        upload_limit: int | None = None,
        receipts_dir: str | None = None,
        confirm: Callable[[ExecutionPlan], bool] = lambda plan: True,
        review: Callable[[ExecutionPlan, Summary], bool] = lambda plan, summary: not summary.over_limit,
) -> Receipt:
    """`review` sees the summary (uploads, sizes) and may decline; by default
    it declines an upload above `upload_limit`. `confirm` is a plain yes/no."""
    environment = EnvironmentCache(plan, filesystem, executor)
    preparer = DataPreparer(plan, filesystem, executor)
    uploader = DataUploader(plan, filesystem, transfer, probe, owner or {})
    receipt = Receipt.create(receipts_dir or receipts.default_receipts_dir(), plan)
    skipped: set[str] = set()       # nodes whose work is already done
    try:
        environment.check_identity()
        env_ready = environment.ready()
        pending_uploads = uploader.pending()
        if pending_uploads and (transfer is None or probe is None):
            raise HpcError("uploads need a file transfer and an owner probe")
        prepare_nodes = [node for node in plan.nodes if node.kind == "prepare"]
        for node in prepare_nodes:
            if preparer.ready(node):
                skipped.add(node.id)
            elif node.location == "login" and not env_ready and \
                    plan.node("setup").location != "login":
                raise HpcError(
                    f"prepare step {node.id} runs on the login node but the environment "
                    "is built by a job; set setup_location: login as well"
                )
        if env_ready and environment.managed:
            skipped.add("setup")

        parent = plan.batch_dir.rsplit("/", 1)[0]
        filesystem.makedirs(parent)
        if not filesystem.mkdir_atomic(plan.batch_dir):
            raise HpcError(f"{plan.batch_dir} already exists; batch directories are never reused")
        receipt.mirror_to(filesystem, f"{plan.batch_dir}/receipt.json")
        stager.stage(plan.sha, f"{plan.batch_dir}/source")
        for relative, content in sorted(plan_files(plan, manifest_raw).items()):
            filesystem.write(f"{plan.batch_dir}/{relative}", content.encode())
        receipt.mark_staged()
    except HpcError as error:
        receipt.finish(receipts.FAILED, str(error))
        raise

    summary = Summary(
        tasks=len(plan.tasks),
        groups=sum(node.kind == "task_set" for node in plan.nodes),
        uploads=[(a.source, a.details["local"], a.details.get("size")) for a in pending_uploads],
        upload_bytes=DataUploader.total_size(pending_uploads),
        upload_limit=upload_limit,
        environment_ready=env_ready,
    )
    if not (review(plan, summary) and confirm(plan)):
        reason = ("upload exceeds upload_limit and was not confirmed"
                  if summary.over_limit else "declined at confirmation")
        receipt.finish(receipts.STAGED, f"not submitted: {reason}")
        return receipt

    try:
        for artifact in pending_uploads:
            uploader.upload(artifact)
        if environment.managed and not env_ready and plan.node("setup").location == "login":
            receipt.node_submitting("setup")
            environment.build_on_login()
            receipt.node_ran("setup")
            skipped.add("setup")
            env_ready = True
        if env_ready:
            run_preflight(plan, executor)
        for node in prepare_nodes:
            if node.id not in skipped and node.location == "login":
                receipt.node_submitting(node.id)
                preparer.run_on_login(node)
                receipt.node_ran(node.id)
                skipped.add(node.id)
        for node_id in skipped:
            if receipt.node(node_id)["status"] == "pending":
                receipt.node_skipped(node_id)
    except HpcError as error:
        receipt.finish(receipts.FAILED, str(error))
        raise

    refs: dict[str, str] = {}
    for node in plan.nodes:
        if node.id in skipped:
            continue
        script = (f"{plan.batch_dir}/groups/{node.group}/{node.id}.sbatch"
                  if node.kind == "task_set" else
                  f"{plan.batch_dir}/nodes/{node.id}/{node.id}.sbatch")
        receipt.node_submitting(node.id)
        try:
            job_ref = submitter.submit(
                node, script, [refs[dep] for dep in node.depends_on if dep in refs])
        except SubmissionUncertain as error:
            receipt.node_unknown(node.id)
            _cancel_recorded(receipt, submitter)
            name = receipt.node(node.id)["job_name"]
            receipt.finish(
                receipts.NEEDS_RECONCILIATION,
                f"submitting {node.id} had an uncertain outcome ({error}); "
                f"check with: squeue --name={name}  (or sacct --name={name})",
            )
            raise HpcError(receipt.data["error"]) from error
        except SubmissionFailed as error:
            receipt.node(node.id)["status"] = "refused"
            _cancel_recorded(receipt, submitter)
            receipt.finish(receipts.FAILED, f"submitting {node.id} failed: {error}")
            raise HpcError(receipt.data["error"]) from error
        refs[node.id] = job_ref
        receipt.node_submitted(node.id, job_ref)
    receipt.finish(receipts.SUBMITTED)
    return receipt
