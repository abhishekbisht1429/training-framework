"""Per-task launch checks: DDP topology against the group's resources, and
the engine options a task may be started with."""

from __future__ import annotations

import math
from collections.abc import Mapping

from training_framework.hpc import framework
from training_framework.hpc.schema import Resources

#: Engine options a manifest may pass, each taking one value. Everything
#: else is refused: `--config`, `--resume-session`, `--extend-session`, and
#: `--override` would make the task run something other than its config,
#: `--debug` bypasses the supervision loop that handles interrupts, and
#: argparse prefix abbreviations would hide either.
ENGINE_OPTIONS = (
    "--heartbeat-timeout",
    "--stop-sync-grace-period",
    "--stop-sync-poll-interval",
    "--process_timeout_on_join",
)
DEFAULT_PROCESS_TIMEOUT_ON_JOIN = 30.0


def check_topology(session: Mapping, resources: Resources) -> list[str]:
    gpus = resources.gpus
    if not framework.is_listed_component(session, "ddp"):
        if gpus > 1:
            return [
                f"requests {gpus} GPUs but configures no 'ddp'; a single "
                "process uses at most one"
            ]
        return []
    ddp = framework.component_entry_of(session, "ddp")
    if not isinstance(ddp, Mapping):
        return ["'ddp' must be configured with a mapping"]
    world_size = ddp.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        return [
            "ddp.world_size must be set explicitly to a positive integer "
            f"(got {world_size!r}); it is never derived from the resources"
        ]
    backend = ddp.get("backend")
    if backend == "nccl":
        if gpus < 1 or world_size != gpus:
            return [
                f"ddp backend 'nccl' needs world_size == gpus >= 1; got "
                f"world_size {world_size}, gpus {gpus}"
            ]
        return []
    if backend not in (None, "gloo"):
        return [f"ddp backend {backend!r} is not supported for HPC batches (nccl, gloo)"]
    problems = []
    if backend is None and gpus >= 1:
        problems.append(
            "the group requests GPUs but ddp.backend is not set (the engine "
            "default is 'gloo'); set it explicitly"
        )
    if world_size > resources.cpus:
        problems.append(
            f"ddp backend 'gloo' runs world_size {world_size} processes but "
            f"the group requests only {resources.cpus} CPUs"
        )
    return problems


def parse_engine_options(options: tuple[str, ...]) -> tuple[float, list[str]]:
    """The `--process_timeout_on_join` value the task will use, and the
    problems with `options`."""
    problems: list[str] = []
    timeout = DEFAULT_PROCESS_TIMEOUT_ON_JOIN
    index = 0
    while index < len(options):
        token = options[index]
        name, _, inline = token.partition("=")
        if name not in ENGINE_OPTIONS:
            problems.append(
                f"training.engine_options: '{token}' is not allowed (allowed: "
                f"{', '.join(ENGINE_OPTIONS)}; config changes belong in "
                "overrides)"
            )
            index += 1
            continue
        if inline:
            value = inline
            index += 1
        elif index + 1 < len(options):
            value = options[index + 1]
            index += 2
        else:
            problems.append(f"training.engine_options: '{name}' needs a value")
            break
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if not math.isfinite(number) or number < 0:
            problems.append(
                f"training.engine_options: '{name}' needs a finite "
                f"non-negative number; got {value!r}"
            )
        elif name == "--process_timeout_on_join":
            timeout = number
    return timeout, problems
