"""The submit service: stage a compiled plan, check it remotely, and submit
its job graph, recording everything in the submission receipt (B11).

Each check runs before the irreversible action that depends on it (B8).
This version supports user-owned environments (`env.existing`) and
`cluster` data; managed environments, uploads, and prepared data arrive
with their artifact services.
"""

from __future__ import annotations

import json
from typing import Callable

from packaging.specifiers import SpecifierSet

from training_framework.hpc import preflight, receipt as receipts
from training_framework.hpc.errors import HpcError
from training_framework.hpc.plan import ExecutionPlan
from training_framework.hpc.ports import (
    RemoteExecutor,
    RemoteFilesystem,
    SchedulerSubmitter,
    SubmissionFailed,
    SubmissionUncertain,
)
from training_framework.hpc.receipt import Receipt
from training_framework.hpc.render import plan_files


class PreflightFailed(HpcError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("preflight failed:\n" + "\n".join(f"- {p}" for p in problems))


def _check_supported(plan: ExecutionPlan) -> None:
    if plan.environment["managed"]:
        raise HpcError(
            "this version submits only with a user-owned environment "
            "(profile env.existing); managed environments are not available yet"
        )
    kinds = sorted({a.kind for a in plan.artifacts if a.kind in ("upload", "prepare")})
    if kinds:
        raise HpcError(
            f"this version cannot submit data sources of kind {', '.join(kinds)} yet"
        )


def run_preflight(plan: ExecutionPlan, executor: RemoteExecutor) -> dict:
    """Facts from the environment, compared with the plan's expectations."""
    task_set = next(node for node in plan.nodes if node.kind == "task_set")
    context = task_set.context
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
        receipts_dir: str | None = None,
        confirm: Callable[[ExecutionPlan], bool] = lambda plan: True,
) -> Receipt:
    _check_supported(plan)
    receipt = Receipt.create(receipts_dir or receipts.default_receipts_dir(), plan)
    try:
        filesystem.makedirs(plan.batch_dir.rsplit("/", 1)[0])
        if not filesystem.mkdir_atomic(plan.batch_dir):
            raise HpcError(f"{plan.batch_dir} already exists; batch directories are never reused")
        receipt.mirror_to(filesystem, f"{plan.batch_dir}/receipt.json")
        stager.stage(plan.sha, f"{plan.batch_dir}/source")
        for relative, content in sorted(plan_files(plan, manifest_raw).items()):
            filesystem.write(f"{plan.batch_dir}/{relative}", content.encode())
        receipt.mark_staged()
        run_preflight(plan, executor)
    except HpcError as error:
        receipt.finish(receipts.FAILED, str(error))
        raise
    if not confirm(plan):
        receipt.finish(receipts.STAGED, "not submitted: declined at confirmation")
        return receipt

    refs: dict[str, str] = {}
    for node in plan.nodes:
        script = (f"{plan.batch_dir}/groups/{node.group}/{node.id}.sbatch"
                  if node.kind == "task_set" else
                  f"{plan.batch_dir}/nodes/{node.id}/{node.id}.sbatch")
        receipt.node_submitting(node.id)
        try:
            job_ref = submitter.submit(node, script, [refs[dep] for dep in node.depends_on])
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
