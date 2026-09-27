"""The session configurations the guides show are built, as written.

Every YAML block of a guide that holds a `sessions` list is a configuration a
reader may copy, so each of its sessions is built here. Only the
`session_config` is replaced (it names the reader's own package and
directories). Components the examples name but the framework does not ship
are registered as the stand-ins below; a block that cannot be built on its
own says so right above it with `<!-- docs-test: skip: <reason> -->`, and is
reported as skipped with that reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from torch import nn

from training_framework.components import (
    Hook,
    ModuleResource,
    Resource,
    Step,
    hook,
    requires_resource,
    resource,
    step,
)
from training_framework.session import TrainingSession

DOCS = Path(__file__).resolve().parents[3] / "docs"
GUIDES = ["guide/04-configuration.md"]

_BLOCK = re.compile(
    r"(?:<!-- docs-test: skip: (?P<skip>[^>]*?) -->\s*\n)?```yaml\n(?P<body>.*?)```",
    re.S,
)


def _examples():
    for guide in GUIDES:
        text = (DOCS / guide).read_text(encoding="utf-8")
        for match in _BLOCK.finditer(text):
            config = yaml.safe_load(match["body"])
            if not isinstance(config, dict) or "sessions" not in config:
                continue
            line = text[:match.start("body")].count("\n")
            for index, session in enumerate(config["sessions"]):
                marks = (
                    [pytest.mark.skip(reason=match["skip"])]
                    if match["skip"] else []
                )
                yield pytest.param(
                    session, id=f"{guide}:{line}:sessions[{index}]", marks=marks,
                )


# -- stand-ins for the reader's own components -------------------------------------


class _Resource(Resource):
    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass


class Model(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.linear(inputs)


class Dataset(_Resource):
    def __len__(self):
        return 0


class Train(Step):
    def run(self, session) -> None:
        return None


@requires_resource("data_manager")
class Evaluator(Hook):
    def pre_session(self, session) -> None:
        pass

    def post_session(self, session) -> None:
        pass


STAND_INS = {
    "model": (resource, Model),
    "classifier": (resource, Model),
    "my_dataset": (resource, Dataset),
    "train": (step, Train),
    "evaluator": (hook, Evaluator),
}


@pytest.fixture(autouse=True)
def _stand_ins():
    for name, (register, component_class) in STAND_INS.items():
        register(name, overwrite=True)(component_class)


@pytest.mark.parametrize("session", list(_examples()))
def test_a_documented_session_builds(tmp_path, session):
    session = dict(session)
    session["session_config"] = {
        "rng_seed": 1,
        "sessions_dir": str(tmp_path),
        "max_iterations": 1,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }

    TrainingSession(session)
