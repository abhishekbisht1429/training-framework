"""Rendering an `ExecutionContext` (building block B4) as shell lines.

Shared by every adapter that runs commands through a shell: scheduler
scripts, the local executor, and (later) SSH. Commands themselves are
passed as argument vectors, never interpolated.
"""

from __future__ import annotations

import shlex

from training_framework.hpc.plan import ExecutionContext


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


def exec_script(context: ExecutionContext) -> str:
    """A `bash -c` script that establishes `context` and then executes its
    positional arguments as the command (`bash -c SCRIPT bash ARGV...`)."""
    return "\n".join(["set -eo pipefail", *render_context(context), 'exec "$@"']) + "\n"
