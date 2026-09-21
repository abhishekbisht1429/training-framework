"""Checkpoint links are checked against what a component was given.

A `ModuleResource` records the prerequisites it asked for and, on restore,
refuses state written by a differently wired instance. Restore calls
`set_state` before `setup`, so a component that first asks in `setup` had
asked for nothing yet and rejected its own valid checkpoint. The check now
compares with the prerequisites the component was given at construction.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pytest
from torch import nn

from tests.test_utils import make_config, resource_named
from training_framework.components import (
    ModuleResource,
    requires_resource,
    resource,
)
from training_framework.session import TrainingSession


def declare_components(*, take_in):
    @resource("link_enc")
    class Encoder(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.linear = nn.Linear(2, 2)

    @requires_resource("link_role")
    @resource("link_model")
    class Model(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.head = nn.Linear(2, 2)
            if take_in == "init":
                self.get_dependency("link_role")

        def setup(self, session) -> None:
            if take_in == "setup":
                self.get_dependency("link_role")


def config_for(tmp_path, bindings, **components):
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = bindings
    config.update(components)
    return config


@pytest.mark.parametrize("take_in", ["init", "setup"])
def test_a_checkpoint_restores_wherever_the_link_was_taken(tmp_path, take_in):
    declare_components(take_in=take_in)
    session = TrainingSession(config_for(
        tmp_path,
        {"link_role": "link_enc"},
        link_enc={},
        link_model={},
    ))
    with session:
        pass
    state = session.get_state()

    restored = TrainingSession.from_state(state)

    with restored:
        pass


def test_state_from_a_differently_wired_instance_is_still_rejected(tmp_path):
    declare_components(take_in="setup")
    session = TrainingSession(config_for(
        tmp_path,
        {
            "link_model#x": {"link_role": "link_enc#a"},
            "link_model#y": {"link_role": "link_enc#b"},
        },
        **{
            "link_enc#a": {},
            "link_enc#b": {},
            "link_model#x": {},
            "link_model#y": {},
        },
    ))
    with session:
        pass
    wired_to_a = resource_named(session, "link_model#x")
    wired_to_b = resource_named(session, "link_model#y")

    with pytest.raises(ValueError, match="checkpointed with linked components"):
        wired_to_b.set_state(wired_to_a.get_state())
