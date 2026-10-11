"""The scheduler composition point.

Everything scheduler-specific comes from one bundle here: how its profile
block is parsed, its pure dialect (compiler), its submitter and owner probe
(submit). Adding a scheduler means adding a bundle; the schema, compiler,
and CLI only look bundles up. The remote, standard-library side has the
matching table in `remote/runtime.py` (`JOB_SCHEDULERS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from training_framework.hpc.errors import HpcError


@dataclass(frozen=True)
class SchedulerBundle:
    name: str
    parse_settings: Callable[..., Any]     # (raw, where, problems, reader_class) -> settings
    dialect: Callable[[Any], Any]          # settings -> SchedulerDialect
    submitter: Callable[[Any], Any]        # RemoteExecutor -> SchedulerSubmitter
    owner_probe: Callable[[Any], Any]      # RemoteExecutor -> LeaseOwnerProbe
    #: RemoteExecutor -> (job refs -> {job or array element id: state}).
    job_states: Callable[[Any], Callable[[list[str]], dict]] | None = None


def _slurm() -> SchedulerBundle:
    from training_framework.hpc.scheduler import slurm, slurm_submitter

    return SchedulerBundle(
        name="slurm",
        parse_settings=slurm.parse_settings,
        dialect=slurm.SlurmDialect,
        submitter=slurm_submitter.SlurmSubmitter,
        owner_probe=slurm_submitter.RemoteOwnerProbe,
        job_states=slurm_submitter.job_states,
    )


class _Registry(dict):
    """Bundles by name, created on first use (no import cycles)."""

    def __init__(self, factories):
        super().__init__()
        self._factories = dict(factories)

    def __iter__(self):
        return iter(self._factories)

    def __contains__(self, name):
        return name in self._factories

    def __getitem__(self, name):
        if not super().__contains__(name):
            if name not in self._factories:
                raise KeyError(name)
            super().__setitem__(name, self._factories[name]())
        return super().__getitem__(name)

    def __setitem__(self, name, bundle):
        self._factories[name] = lambda: bundle
        super().__setitem__(name, bundle)

    def __delitem__(self, name):
        self._factories.pop(name)
        super().pop(name, None)

    def keys(self):
        return self._factories.keys()


SCHEDULERS = _Registry({"slurm": _slurm})


def get_scheduler(name: str) -> SchedulerBundle:
    try:
        return SCHEDULERS[name]
    except KeyError:
        raise HpcError(f"unknown scheduler '{name}' (known: {', '.join(SCHEDULERS)})") from None
