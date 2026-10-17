"""The Slurm `SchedulerDialect`: pure validation and script rendering.

The only module (with the future Slurm submitter and job probe) that knows
`sbatch` directives, arrays, signals, and partitions.
"""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Mapping

from training_framework.hpc.compile.plan import JobNode
from training_framework.hpc.ports.shell import launched, render_context
from dataclasses import dataclass

from training_framework.hpc.config.schema import Resources

#: Directives the dialect sets itself; neither the profile's defaults nor a
#: group's scheduler_options may set them.
OWNED_DIRECTIVES = frozenset({
    "array", "dependency", "kill-on-invalid-dep", "output", "error",
    "signal", "nodes", "ntasks", "ntasks-per-node", "chdir", "job-name",
    "time", "cpus-per-task", "mem", "gres", "gpus", "partition", "account",
    "qos", "parsable", "wrap",
})
#: Owned directives the *profile* (site policy) may still set as defaults.
PROFILE_DIRECTIVES = frozenset({"account", "qos"})


def _unsafe(value) -> bool:
    """Control characters (a newline would start another #SBATCH line)."""
    return any(ord(char) < 32 or ord(char) == 127 for char in str(value))
_DIRECTIVE = re.compile(r"\A[a-z][a-z0-9-]*\Z")


@dataclass(frozen=True)
class SlurmSettings:
    defaults: dict
    partitions: dict
    max_array_size: int


def parse_settings(raw, where, problems, reader_class) -> SlurmSettings:
    """The profile's `slurm:` block."""
    reader = reader_class(raw, where, problems, {"defaults", "partitions", "max_array_size"})
    defaults = reader.get("defaults", {})
    if not isinstance(defaults, Mapping) or not all(
            isinstance(key, str) and key
            and isinstance(value, (str, int)) and not isinstance(value, bool)
            for key, value in defaults.items()
    ):
        problems.add(f"{where}.defaults must map option names (strings) to string "
                     "or integer values")
        defaults = {}
    partitions_reader = reader_class(reader.get("partitions", {}), f"{where}.partitions",
                                     problems, {"gpu", "cpu"})
    partitions = {kind: partitions_reader.string(kind) for kind in ("gpu", "cpu")}
    return SlurmSettings(
        defaults=dict(defaults),
        partitions={kind: name for kind, name in partitions.items() if name is not None},
        max_array_size=reader.positive_int("max_array_size", 1000, required=True),
    )


class SlurmDialect:
    name = "slurm"

    def __init__(self, settings: SlurmSettings):
        self.settings = settings

    @property
    def max_task_set_size(self) -> int:
        return self.settings.max_array_size

    def check_profile(self) -> list[str]:
        problems = []
        for key, value in self.settings.defaults.items():
            if _unsafe(key) or _unsafe(value):
                problems.append(f"profile slurm.defaults: '{key}' contains a control character")
            elif key in PROFILE_DIRECTIVES:
                continue
            elif not _DIRECTIVE.match(key) or key in OWNED_DIRECTIVES:
                problems.append(f"profile slurm.defaults: '{key}' is not allowed")
        for kind, partition in self.settings.partitions.items():
            if _unsafe(partition):
                problems.append(f"profile slurm.partitions.{kind} contains a control character")
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
                elif _unsafe(value):
                    problems.append(
                        f"{where}.scheduler_options.slurm.{key} contains a control character"
                    )
        if resources.gpus > 0 and "gpu" not in self.settings.partitions:
            problems.append(f"{where}: requests GPUs but the profile sets no slurm.partitions.gpu")
        if resources.gpus == 0 and "cpu" not in self.settings.partitions:
            problems.append(f"{where}: requests no GPUs but the profile sets no slurm.partitions.cpu")
        return problems

    def signal_lead(self, process_timeout_on_join: float, stop_grace: int) -> int:
        return int(math.ceil(process_timeout_on_join)) + stop_grace

    def render(self, node: JobNode, batch_id: str, batch_dir: str,
               command: list[str], signal_lead: int | None, *,
               array_indices: list[int] | None = None, resume_table: str | None = None) -> str:
        """`array_indices`: run only these elements of the group's task set
        (a later attempt); `resume_table`: a table mapping each index to the
        checkpoint it resumes from, read into `{checkpoint}`."""
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
            elements = (",".join(str(index) for index in sorted(array_indices))
                        if array_indices else f"0-{node.task_count - 1}")
            directives.append(f"--array={elements}{throttle}")
            directives.append(f"--output={log_dir}/%A_%a.out")
        else:
            directives.append(f"--output={log_dir}/%j.out")
        if signal_lead is not None:
            directives.append(f"--signal=B:USR1@{signal_lead}")
        directives.append(f"--chdir={node.context.working_directory}")

        unsafe = [directive for directive in directives if _unsafe(directive)]
        if unsafe:   # validation refuses these first; never render them
            raise ValueError(f"refusing to render directives with control characters: {unsafe!r}")
        lines = ["#!/bin/bash"]
        lines += [f"#SBATCH {directive}" for directive in directives]
        lines.append("set -eo pipefail")
        lines += render_context(node.context)
        command = launched(node.context, command)
        if node.kind == "task_set":
            lines += _task_body(batch_dir, node.group, command, resume_table)
        else:
            lines.append("exec " + shlex.join(command))
        return "\n".join(lines) + "\n"


def _task_body(batch_dir: str, group: str, command: list[str],
               resume_table: str | None = None) -> list[str]:
    table = shlex.quote(f"{batch_dir}/groups/{group}/tasks.tsv")
    run = shlex.join(command)
    for name in ("config", "record", "checkpoint"):
        run = run.replace(f"'{{{name}}}'", f'"${name}"').replace(f"{{{name}}}", f'"${name}"')
    lookup = []
    if resume_table:
        lookup = [
            f"checkpoint=$(awk -F'\\t' -v i=\"$SLURM_ARRAY_TASK_ID\" "
            f"'NR > 1 && $1 == i {{print $3}}' {shlex.quote(resume_table)})",
            '[ -n "$checkpoint" ] || { echo "no checkpoint for array index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }',
        ]
    return [
        f"config={shlex.quote(batch_dir)}/$(awk -F'\\t' -v i=\"$SLURM_ARRAY_TASK_ID\" "
        f"'NR > 1 && $1 == i {{print $4}}' {table})",
        'record="$(dirname "$config")/record.json"',
        *lookup,
        "interrupted=0",
        "child=''",
        'log() { echo "[hpc $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }',
        'on_signal() { interrupted=1; log "received the scheduler advance signal; stopping the task cooperatively"; '
        '[ -n "$child" ] && kill -INT "$child" 2>/dev/null || true; }',
        "trap on_signal USR1",
        "set +e",
        f"{run} &",
        "child=$!",
        'while true; do wait "$child"; status=$?; kill -0 "$child" 2>/dev/null || break; done',
        'log "task process exited with status $status"',
        'if [ "$interrupted" -eq 1 ]; then exit 75; fi',
        'exit "$status"',
    ]
