"""Batch submission of training experiments to HPC clusters.

The workstation compiles a committed batch manifest and a site profile into
an immutable `ExecutionPlan` (`compiler.compile_plan`); `validate` and
`render` stop there. See `.context/plan-hpc-batch-training.md` for the
protocol this package implements.
"""

from training_framework.hpc.errors import HpcError, PlanError

EXECUTION_PROTOCOL_VERSION = 1
"""The contract between the workstation that compiles a plan and the remote
side that consumes it. Any change to the plan files, record formats, or
wrapper/task arguments bumps it."""

__all__ = ["EXECUTION_PROTOCOL_VERSION", "HpcError", "PlanError"]
