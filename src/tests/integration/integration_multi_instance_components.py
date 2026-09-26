"""Fixtures for running a real session that holds one component twice.

Everything here builds on `integration_training_components`; the only
addition is a stateful hook designed to be configured more than once, so a
real spawned worker has to keep the two instances -- their configuration and
their state -- apart from each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import override

from training_framework.components import (
    ExtendableComponent,
    StatefulLifeCycleHook,
    hook,
)
from training_framework.session import TrainingSession

# Imported for its registrations: the model, dataset and training steps are
# reused as they are.
from tests.integration import integration_training_components  # noqa: F401


@hook("integration_instance_recorder")
class InstanceRecorder(StatefulLifeCycleHook, ExtendableComponent):
    """Record the iterations this instance saw, under its own label.

    Each instance writes its own file, so two of them producing one file
    apiece -- with the right labels and the right names -- is what proves the
    worker kept them apart.
    """

    def __init__(self, config: dict):
        self.call_every = 1
        self._label = config["label"]
        self._output_dir = Path(config["output_dir"])
        self._iterations: list[int] = []

    @override
    def pre_session(self, session: TrainingSession) -> None:
        return None

    @override
    def pre_iteration_callback(self, session: TrainingSession) -> None:
        return None

    @override
    def post_iteration_callback(self, session: TrainingSession) -> None:
        self._iterations.append(session.iteration)

    @override
    def post_session(self, session: TrainingSession) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / f"{self._label}.json").write_text(
            json.dumps({
                "label": self._label,
                "name": self.name,
                "implementation": self.implementation_name,
                "iterations": self._iterations,
            }),
            encoding="utf-8",
        )

    @override
    def apply_extension_config(self, config, changed_paths) -> None:
        unsupported = sorted(
            path for path in changed_paths if path != ("label",)
        )
        if unsupported:
            raise ValueError(
                "Only the label may change during extension; got "
                f"{unsupported}"
            )
        self._label = config["label"]

    @override
    def get_state(self):
        return {"iterations": list(self._iterations)}

    @override
    def set_state(self, state) -> None:
        self._iterations = list(state["iterations"])
