"""Tests for prerequisites recorded as they are handed out.

`get_dependency()` records what it returns, so the framework knows a component
was wired to another one wherever the caller puts the reference. Components are
declared inside the test functions on purpose: the autouse registry fixture
clears the global registries before each test.
"""

import pytest
from torch import nn

from tests.test_utils import build_session, component_named, resource_named
from training_framework.components import (
    LifecycleHook,
    ModuleResource,
    requires_resource,
    hook,
    resource,
)


def _declare_encoder():
    @resource("ld_encoder")
    class Encoder(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.linear = nn.Linear(4, 4)

    return Encoder


def test_a_dependency_is_recorded_wherever_the_caller_puts_it(tmp_path):
    _declare_encoder()

    @requires_resource("ld_encoder")
    @resource("ld_discards")
    class Discards(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            # Consulted for its width, never held.
            encoder = self.get_dependency("ld_encoder")
            self.head = nn.Linear(encoder.linear.out_features, 2)

    session = build_session(tmp_path, {"ld_discards": {}, "ld_encoder": {}})
    discards = resource_named(session, "ld_discards")

    assert discards.linked_components == {"ld_encoder": "ld_encoder"}
    assert set(discards.get_state()["state_dict"]) == {
        "head.weight", "head.bias",
    }
    session.get_state()


def test_a_hook_records_the_dependency_it_asks_for(tmp_path):
    _declare_encoder()

    @requires_resource("ld_encoder")
    @hook("ld_hook")
    class Watcher(LifecycleHook):
        call_every = 1

        def __init__(self, config=None):
            super().__init__(config)
            self.encoder = self.get_dependency("ld_encoder")

        def pre_session(self, session):
            pass

        def post_session(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = build_session(tmp_path, {"ld_hook": {}, "ld_encoder": {}})
    watcher = component_named(session, "ld_hook")

    assert watcher.encoder is resource_named(session, "ld_encoder")
    assert watcher.linked_components == {"ld_encoder": "ld_encoder"}


def _wired_model(tmp_path):
    _declare_encoder()

    @requires_resource("ld_encoder")
    @resource("ld_model")
    class Model(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.encoder = self.get_dependency("ld_encoder")
            self.head = nn.Linear(4, 2)

    session = build_session(tmp_path, {"ld_model": {}, "ld_encoder": {}})
    return resource_named(session, "ld_model")


def test_state_records_the_asked_name_and_the_implementation(tmp_path):
    model = _wired_model(tmp_path)

    state = model.get_state()

    assert state["version"] == 2
    assert state["linked"] == {"ld_encoder": "ld_encoder"}


def test_a_version_1_state_is_read_through_its_implementations(tmp_path):
    model = _wired_model(tmp_path)

    state = model.get_state()
    # Version 1 keyed the map by the attribute the child was attached under,
    # which is no longer knowable.
    state["version"] = 1
    state["linked"] = {"some_old_attribute": "ld_encoder"}

    model.set_state(state)


def test_a_version_1_state_from_a_different_wiring_is_rejected(tmp_path):
    model = _wired_model(tmp_path)

    state = model.get_state()
    state["version"] = 1
    state["linked"] = {"ld_encoder": "something_else"}

    with pytest.raises(ValueError, match="was checkpointed with linked"):
        model.set_state(state)
