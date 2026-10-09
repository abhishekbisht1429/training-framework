"""The Slurm `SchedulerDialect`: pure validation and script rendering.

The only module (with the future Slurm submitter and job probe) that knows
`sbatch` directives, arrays, signals, and partitions.
"""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Mapping

from training_framework.hpc.plan import ExecutionContext, JobNode
from training_framework.hpc.schema import Resources, SlurmSettings

#: Directives the dialect sets itself; neither the profile's defaults nor a
#: group's scheduler_options may set them.
OWNED_DIRECTIVES = frozenset({
    "array", "dependency", "kill-on-invalid-dep", "output", "error",
    "signal", "nodes", "ntasks", "ntasks-per-node", "chdir", "job-name",
    "time", "cpus-per-task", "mem", "gres", "gpus", "partition", "account",
    "parsable", "wrap",
})
_DIRECTIVE = re.compile(r"\A[a-z][a-z0-9-]*\Z")


class SlurmDialect:
    name = "slurm"

    def __init__(self, settings: SlurmSettings):
        self.settings = settings

    @property
    def max_task_set_size(self) -> int:
        return self.settings.max_array_size

    def check_profile(self) -> list[str]:
        problems = []
        for key in self.settings.defaults:
            if key == "account":
                continue
            if not _DIRECTIVE.match(key) or key in OWNED_DIRECTIVES:
                problems.append(f"profile slurm.defaults: '{key}' is not allowed")
        return problems

    def check_resources(self, resources: Resources, where: str) -> list[str]:
        problems = []
        for scheduler, options in resources.scheduler_options.items():
            if scheduler != self.name:
                problems.append(
                    f"{where}.scheduler_options.{scheduler}: options for a "
                    f"scheduler other than the selected '{self.name}'"
                )
                continue
            for key, value in options.items():
                if not isinstance(key, str) or not _DIRECTIVE.match(key):
                    problems.append(f"{where}.scheduler_options.slurm: bad option name {key!r}")
                elif key in OWNED_DIRECTIVES:
                    problems.append(
                        f"{where}.scheduler_options.slurm.{key} is set by the "
                        "HPC layer and cannot be overridden"
                    )
                elif isinstance(value, bool) or not isinstance(value, (str, int)):
                    problems.append(
                        f"{where}.scheduler_options.slurm.{key} must be a string or integer"
                    )
        if resources.gpus > 0 and "gpu" not in self.settings.partitions:
            problems.append(f"{where}: requests GPUs but the profile sets no slurm.partitions.gpu")
        if resources.gpus == 0 and "cpu" not in self.settings.partitions:
            problems.append(f"{where}: requests no GPUs but the profile sets no slurm.partitions.cpu")
        return problems

    def signal_lead(self, process_timeout_on_join: float, stop_grace: int) -> int:
        return int(math.ceil(process_timeout_on_join)) + stop_grace

    def render(self, node: JobNode, batch_id: str, batch_dir: str,
               command: list[str], signal_lead: int | None) -> str:
        # Deterministic, so a job whose reference was lost can be found by
        # name (submission receipt, B11).
        directives = [f"--job-name={batch_id}.{node.id}", "--nodes=1", "--ntasks=1"]
        resources = node.resources
        log_dir = f"{batch_dir}/logs/{node.group or node.id}"
        if resources is not None:
            directives += [
                f"--time={resources.walltime}",
                f"--cpus-per-task={resources.cpus}",
                f"--mem={resources.memory}",
            ]
            if resources.gpus:
                directives.append(f"--gres=gpu:{resources.gpus}")
            partition = self.settings.partitions.get("gpu" if resources.gpus else "cpu")
            if partition:
                directives.append(f"--partition={partition}")
            options = dict(self.settings.defaults)
            options.update(resources.scheduler_options.get(self.name, {}))
            directives += [f"--{key}={value}" for key, value in sorted(options.items())]
        if node.kind == "task_set":
            throttle = f"%{node.max_parallel}" if node.max_parallel else ""
            directives.append(f"--array=0-{node.task_count - 1}{throttle}")
            directives.append(f"--output={log_dir}/%A_%a.out")
        else:
            directives.append(f"--output={log_dir}/%j.out")
        if signal_lead is not None:
            directives.append(f"--signal=B:USR1@{signal_lead}")
        directives.append(f"--chdir={node.context.working_directory}")

        lines = ["#!/bin/bash"]
        lines += [f"#SBATCH {directive}" for directive in directives]
        lines.append("set -euo pipefail")
        lines += render_context(node.context)
        if node.kind == "task_set":
            lines += _task_body(batch_dir, node.group, command)
        else:
            lines.append("exec " + shlex.join(command))
        return "\n".join(lines) + "\n"


def render_context(context: ExecutionContext) -> list[str]:
    """Shell lines that establish an `ExecutionContext`."""
    lines = list(context.hooks)
    if context.unset_variables:
        lines.append("unset " + " ".join(context.unset_variables))
    for name, value in context.set_variables:
        lines.append(f"export {name}={shlex.quote(value)}")
    if context.pythonpath:
        lines.append("export PYTHONPATH=" + shlex.quote(":".join(context.pythonpath)))
    lines.append(f"cd {shlex.quote(context.working_directory)}")
    return lines


def _task_body(batch_dir: str, group: str, command: list[str]) -> list[str]:
    table = shlex.quote(f"{batch_dir}/groups/{group}/tasks.tsv")
    run = shlex.join(command).replace("'{config}'", '"$config"').replace("{config}", '"$config"')
    return [
        f"config={shlex.quote(batch_dir)}/$(awk -F'\\t' -v i=\"$SLURM_ARRAY_TASK_ID\" "
        f"'NR > 1 && $1 == i {{print $4}}' {table})",
        "interrupted=0",
        "child=''",
        'on_signal() { interrupted=1; [ -n "$child" ] && kill -INT "$child" 2>/dev/null || true; }',
        "trap on_signal USR1",
        "set +e",
        f"{run} &",
        "child=$!",
        'while true; do wait "$child"; status=$?; kill -0 "$child" 2>/dev/null || break; done',
        'if [ "$interrupted" -eq 1 ]; then exit 75; fi',
        'exit "$status"',
    ]
