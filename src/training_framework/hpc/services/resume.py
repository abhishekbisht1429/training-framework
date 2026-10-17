"""`resume`: continue a batch's unfinished tasks from their checkpoints, as a
new attempt in the same batch directory.

The engine already resumes a session exactly (`--resume-session`: weights,
optimizer, RNG, data position, and the sampler seed saved in the
checkpoint); this module decides which tasks to resume and from which
checkpoint, checks the cluster can still run them, renders one job array
per group with the same environment and task wrapper (so the early-stop
signal, logs, and records work as in the first attempt), and submits them
under the receipt's transactional rules (building block B11).

Attempt 1 is the original submission; later attempts are recorded in the
receipt's `attempts` list and as nodes `group-<group>.resume-<n>` carrying
`attempt`, `group`, and the array `tasks` they ran.
"""

from __future__ import annotations

import base64
import json
import posixpath
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml

from training_framework.hpc.compile.plan import ExecutionContext, JobNode
from training_framework.hpc.config.schema import Resources, parse_size, parse_walltime
from training_framework.hpc.config.topology import parse_engine_options
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import SubmissionFailed, SubmissionUncertain
from training_framework.hpc.remote import preflight
from training_framework.hpc.remote.lease import KeyMismatch, check_marker
from training_framework.hpc.services.status import TERMINAL_STATES, Row, is_task

#: Task states resumed by default; failed tasks only when named or asked for.
DEFAULT_STATES = ("interrupted", "ended without a record")


class ResumeError(HpcError):
    pass


@dataclass
class Candidate:
    task: str
    group: str
    index: int
    state: str
    checkpoint: str | None = None       # absolute path on the cluster
    iteration: int | None = None
    max_iterations: int | None = None
    reason: str | None = None           # why it cannot be resumed


@dataclass
class GroupAttempt:
    group: str
    node: JobNode
    tasks: list[Candidate]
    table_path: str
    table: str
    script_path: str
    script: str = ""


@dataclass
class ResumePlan:
    attempt: int
    chosen: list[Candidate]
    skipped: list[Candidate]
    groups: list[GroupAttempt] = field(default_factory=list)


# Selection ------------------------------------------------------------------------


def task_index(tables: dict[str, str]) -> dict[str, tuple[str, int]]:
    """Task name -> (group, array index), from the groups' task tables."""
    found = {}
    for group, table in tables.items():
        for line in table.splitlines()[1:]:
            index, task = line.split("\t")[:2]
            found[task] = (group, int(index))
    return found


def active_jobs(receipt: dict, live: dict) -> list[str]:
    """This batch's jobs (any attempt) still in the scheduler's queue."""
    refs = [node["job_ref"] for node in receipt["nodes"].values() if node.get("job_ref")]
    return sorted(element for element, state in live.items() if state not in TERMINAL_STATES
                  and any(element == ref or element.startswith(f"{ref}_") for ref in refs))


def select_tasks(receipt: dict, rows: list[Row], index: dict[str, tuple[str, int]],
                 names: list[str] | None = None, include_failed: bool = False) -> list[Candidate]:
    """The tasks to resume: the named ones, or every task in DEFAULT_STATES
    (and failed ones with `include_failed`). A finished task is never resumed."""
    states = {row.name: row.state for row in rows if is_task(row, receipt) and row.name in index}
    if names:
        problems = []
        for name in names:
            if name not in states:
                problems.append(f"'{name}' is not a task of this batch")
            elif states[name] == "done":
                problems.append(f"'{name}' finished (done); a finished task is not resumed")
        if problems:
            raise ResumeError("cannot resume: " + "; ".join(problems))
        chosen = list(dict.fromkeys(names))
    else:
        chosen = [name for name, state in states.items()
                  if state in DEFAULT_STATES or (include_failed and state.startswith("failed"))]
    return [Candidate(name, *index[name], states[name]) for name in chosen]


# Checkpoints ----------------------------------------------------------------------

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
        raise ResumeError(f"cannot list checkpoints in {batch_dir}: {result.stderr.strip()}")
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


def _has_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


def _max_iterations(config: dict | None) -> int | None:
    try:
        return int(config["sessions"][0]["session_config"]["max_iterations"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def choose_checkpoints(candidates: list[Candidate], checkpoints, configs,
                       overrides: dict[str, tuple[str, int | None]] | None = None) -> None:
    """Fill each candidate's checkpoint (the newest complete one, by
    iteration), or the reason it cannot be resumed."""
    overrides = overrides or {}
    for candidate in candidates:
        config = configs.get(candidate.task)
        candidate.max_iterations = _max_iterations(config)
        if candidate.task in overrides:
            candidate.checkpoint, candidate.iteration = overrides[candidate.task]
        else:
            found = [item for item in checkpoints.get(candidate.task, []) if item[1] is not None]
            if not found:
                session = (config or {}).get("sessions", [{}])[0] if config else {}
                if not _has_key(session, "checkpointer"):
                    candidate.reason = "no checkpoint: the task config has no checkpointer"
                elif _has_key(session, "checkpoints_dir"):
                    candidate.reason = ("no checkpoint under the task's sessions directory (an "
                                        "explicit checkpoints_dir is not searched; pass "
                                        "--checkpoint NAME=PATH)")
                else:
                    candidate.reason = "no complete checkpoint was written yet"
                continue
            candidate.checkpoint, candidate.iteration = max(found, key=lambda item: (item[1], item[0]))
        if (candidate.max_iterations is not None and candidate.iteration is not None
                and candidate.iteration >= candidate.max_iterations):
            candidate.reason = (f"its checkpoint is at iteration {candidate.iteration}, the "
                                "configured max_iterations: nothing is left to run")


def explicit_checkpoints(filesystem, batch_dir: str, given: list[str]) -> dict[str, tuple[str, int | None]]:
    """`NAME=PATH` arguments: each a complete checkpoint inside the batch."""
    chosen = {}
    for item in given:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ResumeError(f"--checkpoint {item!r} must be NAME=PATH")
        path = path if posixpath.isabs(path) else f"{batch_dir}/{path}"
        path = posixpath.normpath(path)
        if not path.startswith(batch_dir.rstrip("/") + "/"):
            raise ResumeError(f"--checkpoint {item!r}: {path} is outside the batch directory")
        if path.endswith(".tmp"):
            raise ResumeError(f"--checkpoint {item!r}: {path} is an incomplete checkpoint")
        manifest = filesystem.read(f"{path}/manifest.json")
        if manifest is None:
            raise ResumeError(f"--checkpoint {item!r}: {path} is not a complete checkpoint "
                              "(no manifest.json)")
        try:
            iteration = json.loads(manifest).get("iteration")
        except ValueError:
            raise ResumeError(f"--checkpoint {item!r}: {path}/manifest.json is not valid JSON") from None
        chosen[name] = (path, iteration if isinstance(iteration, int) else None)
    return chosen


# The cluster side --------------------------------------------------------------------


def context_from_json(data: dict) -> ExecutionContext:
    return ExecutionContext(
        phase=data["phase"], interpreter=data["interpreter"], environment=data["environment"],
        working_directory=data["working_directory"], pythonpath=tuple(data["pythonpath"]),
        set_variables=tuple(tuple(item) for item in data["set_variables"]),
        unset_variables=tuple(data["unset_variables"]), hooks=tuple(data["hooks"]),
        path_prepend=tuple(data.get("path_prepend", ())), launcher=tuple(data.get("launcher", ())),
    )


def read_plan(filesystem, batch_dir: str) -> dict:
    data = filesystem.read(f"{batch_dir}/plan.json")
    if data is None:
        raise ResumeError(f"{batch_dir}/plan.json is missing on the cluster")
    return json.loads(data)


def plan_dialect(plan: dict):
    """The scheduler dialect with the settings the batch was rendered with
    (account, QoS, partitions...), not the profile's current ones."""
    from training_framework.hpc.config.schema import _Reader
    from training_framework.hpc.errors import Problems
    from training_framework.hpc.scheduler import get_scheduler

    raw = plan.get("scheduler_settings")
    if not raw:
        raise ResumeError(f"{plan['batch_id']} was submitted before batches recorded what resume "
                          "needs (its scheduler settings in plan.json); submit a new batch")
    bundle = get_scheduler(plan["scheduler"])
    problems = Problems()
    settings = bundle.parse_settings(raw, "plan.json scheduler_settings", problems, _Reader)
    if problems:
        raise ResumeError(f"plan.json scheduler settings are not valid: {problems}")
    return bundle.dialect(settings)


def check_cluster(plan: dict, filesystem, executor) -> None:
    """The source, the environment, and a remote side that can resume."""
    batch_dir = plan["batch_dir"]
    if not filesystem.exists(f"{batch_dir}/source"):
        raise ResumeError(f"the batch's source checkout {batch_dir}/source is gone; submit a new batch")
    environment = plan["environment"]
    if environment.get("managed"):
        try:
            ready = check_marker(filesystem, environment["location"], environment["key"])
        except KeyMismatch as error:
            raise ResumeError(str(error)) from None
        if not ready:
            raise ResumeError(f"the batch's environment {environment['location']} is no longer "
                              "ready (deleted or purged); submit a new batch")
    group = next(node for node in plan["nodes"] if node["kind"] == "task_set")
    context = context_from_json(group["context"])
    result = executor.run([context.interpreter, "-m", "training_framework.hpc.task",
                           "--preflight", f"{batch_dir}/plan.json"], context)
    try:
        facts = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ResumeError("the environment cannot run training_framework.hpc.task "
                          f"(exit {result.status}): {result.stderr.strip()[-2000:]}") from None
    problems = preflight.compare(facts, plan["preflight"], f"{batch_dir}/source")
    if "resume" not in (facts.get("capabilities") or []):
        problems.append("the batch's training_framework.hpc predates resume (its task entry "
                        "point has no --resume); submit a new batch with a newer framework")
    if problems:
        raise ResumeError("the cluster cannot resume this batch:\n"
                          + "\n".join(f"- {problem}" for problem in problems))


# Rendering ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceChanges:
    walltime: str | None = None
    cpus: int | None = None
    memory: str | None = None
    max_parallel: int | None = None

    def check(self) -> None:
        if self.walltime is not None and parse_walltime(self.walltime) is None:
            raise ResumeError(f"--walltime must be [D-]HH:MM:SS; got {self.walltime!r}")
        if self.memory is not None and parse_size(self.memory) is None:
            raise ResumeError(f"--memory must be a size such as 64G; got {self.memory!r}")
        for name in ("cpus", "max_parallel"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ResumeError(f"--{name.replace('_', '-')} must be at least 1")


def build_attempt(plan: dict, receipt: dict, chosen: list[Candidate], dialect,
                  changes: ResourceChanges) -> ResumePlan:
    """One array per group with chosen tasks, rendered like the first attempt."""
    changes.check()
    attempt = 1 + max([item.get("attempt", 1) for item in receipt.get("attempts", [])] or [1])
    timeout, _ = parse_engine_options(tuple(plan["engine_options"]))
    batch_id, batch_dir = plan["batch_id"], plan["batch_dir"]
    result = ResumePlan(attempt, chosen, [])
    by_group: dict[str, list[Candidate]] = {}
    for candidate in chosen:
        by_group.setdefault(candidate.group, []).append(candidate)
    for group, tasks in sorted(by_group.items()):
        node_json = next((n for n in plan["nodes"] if n["id"] == f"group-{group}"), None)
        if node_json is None or not node_json.get("resources"):
            raise ResumeError(f"{batch_id} was submitted before batches recorded what resume "
                              "needs (group resources in plan.json); submit a new batch")
        raw = node_json["resources"]
        walltime = changes.walltime or raw["walltime"]
        resources = Resources(
            walltime=walltime, walltime_seconds=parse_walltime(walltime),
            gpus=raw["gpus"], cpus=changes.cpus or raw["cpus"],
            memory=changes.memory or raw["memory"],
            scheduler_options=raw.get("scheduler_options") or {},
        )
        problems = dialect.check_resources(resources, f"group '{group}'")
        if problems:
            raise ResumeError("; ".join(problems))
        stop_grace = node_json.get("stop_grace_seconds", 0)
        lead = dialect.signal_lead(timeout, stop_grace)
        if lead >= resources.walltime_seconds:
            raise ResumeError(f"group '{group}': the shutdown signal lead ({lead}s) is not "
                              f"shorter than the walltime ({walltime})")
        node = JobNode(
            id=f"group-{group}.resume-{attempt}", kind="task_set", location="job",
            depends_on=(), context=context_from_json(node_json["context"]),
            resources=resources, group=group, task_count=len(tasks),
            max_parallel=changes.max_parallel or node_json.get("max_parallel"),
            stop_grace_seconds=stop_grace,
        )
        table_path = f"{batch_dir}/groups/{group}/resume-{attempt}.tsv"
        table = "\n".join(["index\ttask\tcheckpoint\titeration"] + [
            f"{task.index}\t{task.task}\t{task.checkpoint}\t{task.iteration}" for task in tasks
        ]) + "\n"
        command = [
            node.context.interpreter, "-m", "training_framework.hpc.task",
            "--config", "{config}", "--record", "{record}", "--plan", f"{batch_dir}/plan.json",
            "--resume", "{checkpoint}", "--attempt", str(attempt), *plan["engine_options"],
        ]
        script = dialect.render(node, batch_id, batch_dir, command, lead,
                                array_indices=[task.index for task in tasks],
                                resume_table=table_path)
        result.groups.append(GroupAttempt(
            group, node, tasks, table_path, table,
            f"{batch_dir}/groups/{group}/{node.id}.sbatch", script,
        ))
    return result


# Submission ---------------------------------------------------------------------------


def submit_attempt(receipt, plan: ResumePlan, filesystem, submitter, *,
                   submitted_from: str | None = None, progress=lambda message: None) -> dict:
    """Stage and submit the attempt; record it in `receipt` (a `Receipt`
    mirrored to the cluster). Returns the attempt's entry. Follows B11: the
    receipt is written before and after each submission; a definite failure
    cancels this attempt's earlier jobs; an uncertain outcome needs
    reconciliation; Ctrl-C cancels what this attempt submitted."""
    entry = {
        "attempt": plan.attempt,
        "created": datetime.now(timezone.utc).isoformat(),
        "submitted_from": submitted_from,
        "resources": {g.group: {"walltime": g.node.resources.walltime, "cpus": g.node.resources.cpus,
                                "memory": g.node.resources.memory, "max_parallel": g.node.max_parallel}
                      for g in plan.groups},
        "tasks": {task.task: {"group": task.group, "index": task.index,
                              "checkpoint": task.checkpoint, "iteration": task.iteration}
                  for task in plan.chosen},
        "nodes": [g.node.id for g in plan.groups],
        "status": "submitting",
        "error": None,
    }
    # The table and script go to the cluster first: if writing them fails,
    # nothing is recorded and nothing was submitted (a later resume reuses
    # this attempt number and overwrites them).
    try:
        for g in plan.groups:
            filesystem.write(g.table_path, g.table.encode())
            filesystem.write(g.script_path, g.script.encode())
    except Exception as error:
        raise ResumeError(f"could not write attempt {plan.attempt}'s job scripts to the cluster: "
                          f"{type(error).__name__}: {error}; nothing was submitted") from error
    data = receipt.data
    data.setdefault("attempts", []).append(entry)
    for g in plan.groups:
        data["nodes"][g.node.id] = {
            "status": "pending", "job_name": f"{data['batch_id']}.{g.node.id}", "job_ref": None,
            "depends_on": [], "cancel": None,
            "attempt": plan.attempt, "group": g.group, "tasks": [task.index for task in g.tasks],
        }
    receipt.save()

    submitted: list[tuple[str, str]] = []

    def cancel_submitted():
        for node_id, job_ref in reversed(submitted):
            try:
                outcome = submitter.cancel(job_ref)
                receipt.node_cancelled(node_id, outcome.ok, outcome.detail)
            except Exception as error:  # recorded, never swallowed
                receipt.node_cancelled(node_id, False, f"{type(error).__name__}: {error}")

    def finish(status: str, error: str | None = None):
        entry["status"], entry["error"] = status, error
        receipt.save()

    current = None
    try:
        for g in plan.groups:
            current = g.node.id
            receipt.node_submitting(current)
            job_ref = submitter.submit(g.node, g.script_path, [])
            receipt.node_submitted(current, job_ref)
            submitted.append((current, job_ref))
            progress(f"submitted {current} as job {job_ref} ({len(g.tasks)} task(s))")
            current = None
    except SubmissionUncertain as error:
        receipt.node_unknown(current)
        cancel_submitted()
        name = data["nodes"][current]["job_name"]
        finish("needs reconciliation", f"submitting {current} had an uncertain outcome ({error}); "
                                       f"check with: squeue --name={name}")
        raise ResumeError(entry["error"]) from error
    except SubmissionFailed as error:
        data["nodes"][current]["status"] = "refused"
        cancel_submitted()
        finish("failed", f"submitting {current} failed: {error}")
        raise ResumeError(entry["error"]) from error
    except Exception as error:
        # Anything else (a transport error, a bug): the sbatch in flight may
        # or may not have been accepted.
        if current is not None:
            receipt.node_unknown(current)
        cancel_submitted()
        where = f"submitting {current}" if current else "the attempt"
        name = data["nodes"][current]["job_name"] if current else None
        finish("needs reconciliation" if current else "failed",
               f"{where} failed unexpectedly ({type(error).__name__}: {error})"
               + (f"; check with: squeue --name={name}" if name else ""))
        raise ResumeError(entry["error"]) from error
    except KeyboardInterrupt:
        if current is not None:
            receipt.node_unknown(current)
        cancel_submitted()
        if current is not None:
            finish("needs reconciliation", f"interrupted while submitting {current}; check with: "
                                           f"squeue --name={data['nodes'][current]['job_name']}")
        else:
            finish("failed", "interrupted (Ctrl-C); no jobs of this attempt left submitted")
        raise
    finish("submitted")
    return entry
