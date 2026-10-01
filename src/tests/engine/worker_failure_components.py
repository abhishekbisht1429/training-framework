"""Components that make a spawned worker fail in each way the parent sees.

Imported by the worker through `components_package`, so they live in a
module of their own rather than in the test.
"""

from __future__ import annotations

import os
import signal
from typing import override

from training_framework.components import SessionHook, Step, hook, step
from training_framework.session import Session, TrainingSession


@step("wf_raise")
class RaisingStep(Step):
    """Raise inside the session: reported by the session's own exit."""

    def __init__(self, config: dict):
        self.config = dict(config)

    @override
    def run(self, session: TrainingSession) -> None:
        raise RuntimeError(self.config.get("message", "raised in a step"))


@hook("wf_raise_on_setup")
class RaisingSetupHook(SessionHook):
    """Raise while the session is entered: reported by the worker loop."""

    @override
    def pre_session(self, session: Session) -> None:
        raise ValueError("raised while entering the session")

    @override
    def post_session(self, session: Session) -> None:
        pass


@step("wf_exit")
class ExitingStep(Step):
    """End the process without raising: nothing is reported."""

    def __init__(self, config: dict):
        self.config = dict(config)

    @override
    def run(self, session: TrainingSession) -> None:
        os._exit(int(self.config.get("code", 3)))


@step("wf_kill")
class KilledStep(Step):
    """Die by SIGKILL, as the out-of-memory killer would."""

    @override
    def run(self, session: TrainingSession) -> None:
        os.kill(os.getpid(), signal.SIGKILL)
