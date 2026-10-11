"""The `ExecutionPlan`: everything a batch will do, decided on the workstation.

Immutable and scheduler-neutral, except for the job scripts the dialect
rendered into it. `as_json` is canonical, so equal inputs give equal plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from training_framework.hpc.names import canonical_json


@dataclass(frozen=True)
class ExecutionContext:
    """What one remote run sees (building block B4)."""

    phase: str                     # setup | prepare | task
    interpreter: str
    environment: str | None
    working_directory: str
    pythonpath: tuple[str, ...]
    set_variables: tuple[tuple[str, str], ...]
    unset_variables: tuple[str, ...]
    hooks: tuple[str, ...]
    #: Directories put first on PATH (the environment's `bin`), so commands
    #: such as a prepare step's `python` resolve inside the environment.
    path_prepend: tuple[str, ...] = ()
    #: Argument vector every command runs under (e.g. `conda run -p ENV`).
    launcher: tuple[str, ...] = ()

    def as_json(self) -> dict:
        return {
            "path_prepend": list(self.path_prepend),
            "launcher": list(self.launcher),
            "phase": self.phase,
            "interpreter": self.interpreter,
            "environment": self.environment,
            "working_directory": self.working_directory,
            "pythonpath": list(self.pythonpath),
            "set_variables": [list(item) for item in self.set_variables],
            "unset_variables": list(self.unset_variables),
            "hooks": list(self.hooks),
        }


@dataclass(frozen=True)
class Task:
    group: str
    experiment: str
    number: int                    # NNN within the experiment
    array_index: int               # index within the group's task set
    variant: tuple[dict, ...]      # [{address, value}] of the matrix
    config: dict = field(repr=False)  # the full training config file
    config_hash: str = ""

    @property
    def name(self) -> str:
        return f"{self.group}/{self.experiment}/{self.number:03d}"

    @property
    def directory(self) -> str:
        """Relative to the batch directory."""
        return f"tasks/{self.name}"


@dataclass(frozen=True)
class Artifact:
    kind: str                      # environment | upload | prepare | cluster
    source: str                    # environment, or the data source name
    key: str | None
    location: str
    location_kind: str             # key-derived | user-chosen | given
    details: dict

    def as_json(self) -> dict:
        return {
            "kind": self.kind, "source": self.source, "key": self.key,
            "location": self.location, "location_kind": self.location_kind,
            "details": self.details,
        }


@dataclass(frozen=True)
class JobNode:
    id: str
    kind: str                      # setup | prepare | task_set
    location: str                  # job | login
    depends_on: tuple[str, ...]
    context: ExecutionContext
    resources: Any = None          # schema.Resources for scheduled nodes
    group: str | None = None
    task_count: int = 0
    max_parallel: int | None = None
    stop_grace_seconds: int = 0
    artifact: str | None = None    # the artifact a setup/prepare node builds
    script: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    protocol_version: int
    batch_id: str
    batch_dir: str
    sha: str
    manifest_hash: str
    profile: str
    scheduler: str
    code_mode: str
    tasks: tuple[Task, ...]
    artifacts: tuple[Artifact, ...]
    nodes: tuple[JobNode, ...]
    environment: dict
    preflight: dict
    engine_options: tuple[str, ...]
    notes: tuple[str, ...]

    def node(self, node_id: str) -> JobNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def as_json(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "batch_id": self.batch_id,
            "batch_dir": self.batch_dir,
            "sha": self.sha,
            "manifest_hash": self.manifest_hash,
            "profile": self.profile,
            "scheduler": self.scheduler,
            "code_mode": self.code_mode,
            "environment": self.environment,
            "preflight": self.preflight,
            "engine_options": list(self.engine_options),
            "notes": list(self.notes),
            "artifacts": [artifact.as_json() for artifact in self.artifacts],
            "nodes": [
                {
                    "id": node.id, "kind": node.kind, "location": node.location,
                    "depends_on": list(node.depends_on),
                    "context": node.context.as_json(),
                    "group": node.group, "task_count": node.task_count,
                    "max_parallel": node.max_parallel,
                    "artifact": node.artifact,
                }
                for node in self.nodes
            ],
            "tasks": [
                {
                    "name": task.name, "group": task.group,
                    "experiment": task.experiment, "number": task.number,
                    "array_index": task.array_index,
                    "variant": list(task.variant),
                    "config_hash": task.config_hash,
                }
                for task in self.tasks
            ],
        }

    def canonical(self) -> str:
        return canonical_json(self.as_json())
