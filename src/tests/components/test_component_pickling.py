"""A component pickled on its own leaves its session behind.

What a session hands a component -- its prerequisites, and the record of
which ones it has asked for -- belongs to that session. A component pickled
on its own must not carry copies of them, whatever order its bases are
listed in.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

from tests.test_utils import build_session, component_named, resource_named
from training_framework.components import (
    ModuleResource,
    Resource,
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)


def _register_unpicklable_source():
    """`pk_source`: a prerequisite that cannot be pickled (its class is
    local), so any pickle that carries it fails."""

    @resource("pk_source", overwrite=True)
    class Source(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass


@requires_resource("pk_source")
class AskingStep(Step):
    """Asks for its prerequisite, and keeps nothing of it."""

    def __init__(self, config=None):
        super().__init__(config)
        self.get_dependency("pk_source")

    def run(self, session):
        pass


@requires_resource("pk_source")
class ModuleFirst(nn.Module, Resource):
    """A module listed before its component base, given a prerequisite."""

    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.linear = nn.Linear(2, 1)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("pk_source")
class StatefulModuleFirst(nn.Module, StatefulResource):
    """A stateful module listed before its component base."""

    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.register_buffer("count", torch.zeros(()))

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return {"count": self.count.clone()}

    def set_state(self, state):
        self.count.copy_(state["count"])


def _pickled(component):
    return pickle.loads(pickle.dumps(component))


def test_a_step_that_asked_for_a_prerequisite_pickles_without_it(tmp_path):
    _register_unpicklable_source()
    step("pk_step", overwrite=True)(AskingStep)
    session = build_session(tmp_path, {"pk_source": {}, "pk_step": {}})
    original = component_named(session, "pk_step")
    assert original.linked_components == {"pk_source": "pk_source"}

    restored = _pickled(original)

    assert restored.linked_components == {}


def test_a_module_listed_first_pickles_without_its_prerequisites(tmp_path):
    _register_unpicklable_source()
    resource("pk_module", overwrite=True)(ModuleFirst)
    session = build_session(tmp_path, {"pk_source": {}, "pk_module": {}})
    original = resource_named(session, "pk_module")

    restored = _pickled(original)

    torch.testing.assert_close(restored.linear.weight, original.linear.weight)


def test_a_stateful_module_listed_first_keeps_its_name_and_state(tmp_path):
    _register_unpicklable_source()
    resource("pk_counter", overwrite=True)(StatefulModuleFirst)
    session = build_session(tmp_path, {"pk_source": {}, "pk_counter#a": {}})
    original = resource_named(session, "pk_counter#a")
    original.count.fill_(5)

    restored = _pickled(original)

    assert restored.name == "pk_counter#a"
    assert restored.count.item() == 5


class OwnProtocolModule(nn.Module):
    """A module base with a pickle protocol of its own, both methods."""

    def __getstate__(self):
        return {"own": self.__dict__.copy()}

    def __setstate__(self, state):
        self.__dict__.update(state["own"])


@requires_resource("pk_source")
class OwnProtocolComponent(OwnProtocolModule, Resource):
    def __init__(self, config=None):
        OwnProtocolModule.__init__(self)
        self.linear = nn.Linear(2, 1)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class DropsScratchModule(nn.Module):
    """A module base defining only `__getstate__`: it drops `scratch`."""

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("scratch", None)
        return state


class ScratchComponent(DropsScratchModule, Resource):
    def __init__(self, config=None):
        DropsScratchModule.__init__(self)
        self.scratch = "not pickled"

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def test_a_base_with_its_own_pickle_protocol_keeps_it_whole(tmp_path):
    _register_unpicklable_source()
    resource("pk_own", overwrite=True)(OwnProtocolComponent)
    session = build_session(tmp_path, {"pk_source": {}, "pk_own": {}})
    original = resource_named(session, "pk_own")

    restored = _pickled(original)

    torch.testing.assert_close(restored.linear.weight, original.linear.weight)
    assert not restored.has_dependency("pk_source")


def test_a_base_defining_only_getstate_keeps_it():
    restored = _pickled(ScratchComponent())

    assert not hasattr(restored, "scratch")


@requires_resource("pk_encoder")
class ComposedAtConstruction(ModuleResource):
    """Takes its prerequisite in the constructor: cannot be rebuilt alone."""

    def __init__(self, config=None):
        super().__init__(config)
        self.encoder = self.get_dependency("pk_encoder")


@requires_resource("pk_encoder")
class ComposedAtSetup(ModuleResource):
    """Takes its prerequisite only once it is set up."""

    def __init__(self, config=None):
        super().__init__(config)
        self.head = nn.Linear(2, 1)

    def setup(self, session):
        super().setup(session)
        self.get_dependency("pk_encoder")


class Encoder(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(2, 2)


def test_a_component_built_from_its_prerequisites_refuses_to_pickle_alone(tmp_path):
    resource("pk_encoder", overwrite=True)(Encoder)
    resource("pk_composed", overwrite=True)(ComposedAtConstruction)
    session = build_session(tmp_path, {"pk_encoder": {}, "pk_composed": {}})

    with pytest.raises(TypeError, match=r"pk_composed .*constructor.*pk_encoder"):
        pickle.dumps(resource_named(session, "pk_composed"))


def test_a_component_taking_prerequisites_later_pickles_alone(tmp_path):
    resource("pk_encoder", overwrite=True)(Encoder)
    resource("pk_composed", overwrite=True)(ComposedAtSetup)
    session = build_session(tmp_path, {"pk_encoder": {}, "pk_composed": {}})
    original = resource_named(session, "pk_composed")

    restored = _pickled(original)

    assert restored.name == "pk_composed"
    torch.testing.assert_close(restored.head.weight, original.head.weight)


def test_a_component_registered_elsewhere_starts_with_nothing_asked_for(tmp_path):
    _register_unpicklable_source()
    step("pk_step", overwrite=True)(AskingStep)
    first = build_session(tmp_path, {"pk_source": {}, "pk_step": {}})
    moved = component_named(first, "pk_step")
    first.remove_step("pk_step")
    second = build_session(tmp_path, {"pk_source": {}})

    second.add_step(moved)

    assert moved.linked_components == {}
    assert moved.has_dependency("pk_source")
