"""A component pickled on its own leaves its session behind.

What a session hands a component -- its prerequisites, and the record of
which ones it has asked for -- belongs to that session. A component pickled
on its own must not carry copies of them, whatever order its bases are
listed in.
"""

from __future__ import annotations

import copy
import copyreg
import gc
import pickle
import weakref

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
    """Takes its prerequisite in the constructor and keeps it as a
    submodule: part of the module, so it travels with it."""

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


def test_a_module_resource_built_from_its_prerequisites_pickles_with_its_parts(tmp_path):
    # A ModuleResource pickles as a module: the part it took in its
    # constructor is its submodule and comes along; its session does not.
    resource("pk_encoder", overwrite=True)(Encoder)
    resource("pk_composed", overwrite=True)(ComposedAtConstruction)
    session = build_session(tmp_path, {"pk_encoder": {}, "pk_composed": {}})
    original = resource_named(session, "pk_composed")

    restored = _pickled(original)

    torch.testing.assert_close(
        restored.encoder.linear.weight, original.encoder.linear.weight,
    )
    assert not restored.has_dependency("pk_encoder")


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


# -- the holder belongs to its owner; copies and cycles ------------------------------


class SimpleSource(Resource):
    """A prerequisite that copies fine (module level)."""

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def _asking_step_session(tmp_path):
    resource("pk_source", overwrite=True)(SimpleSource)
    step("pk_step", overwrite=True)(AskingStep)
    return build_session(tmp_path, {"pk_source": {}, "pk_step": {}})


@pytest.mark.parametrize("duplicate", [copy.copy, copy.deepcopy])
def test_a_copy_of_a_component_leaves_its_session_behind(tmp_path, duplicate):
    original = component_named(_asking_step_session(tmp_path), "pk_step")

    copied = duplicate(original)

    assert not copied.has_dependency("pk_source")
    assert copied.linked_components == {}
    assert original.has_dependency("pk_source")
    assert original.linked_components == {"pk_source": "pk_source"}


def _copy_on_a_dead_original(tmp_path):
    """A copy of a copy that sits where the original component was: same
    id, original collected. None if the allocator did not reuse the spot."""
    session = _asking_step_session(tmp_path)
    original = component_named(session, "pk_step")
    original_id = id(original)
    first = copy.copy(original)
    del session, original
    gc.collect()
    copies = [copy.copy(first) for _ in range(50)]
    return next((c for c in copies if id(c) == original_id), None)


def test_a_copy_at_the_address_of_its_collected_original_has_no_session(tmp_path):
    # An id would say this copy is the holder's owner; only a reference to
    # the owner itself tells them apart.
    for attempt in range(20):
        on_the_spot = _copy_on_a_dead_original(tmp_path / str(attempt))
        if on_the_spot is not None:
            break
    else:
        pytest.skip("the allocator never reused the original's address")

    assert not on_the_spot.has_dependency("pk_source")
    assert on_the_spot.linked_components == {}


def test_a_consumer_its_prerequisite_refers_back_to_is_collected(tmp_path):
    session = _asking_step_session(tmp_path)
    consumer = component_named(session, "pk_step")
    source = resource_named(session, "pk_source")
    source.consumer = consumer
    alive = [weakref.ref(consumer), weakref.ref(source)]

    del session, consumer, source
    gc.collect()

    assert [ref() for ref in alive] == [None, None]


# -- a rebuilt component whose constructor looked at its wiring -----------------------


@requires_resource("pk_source")
class ChecksWiring(StatefulResource):
    """Its constructor only asks whether a prerequisite is there."""

    asked_about = "pk_source"

    def __init__(self, config=None):
        super().__init__(config)
        self.wired = self.has_dependency(type(self).asked_about)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return {}

    def set_state(self, state):
        pass


class ChecksUndeclaredWiring(ChecksWiring):
    asked_about = "pk_other"


@requires_resource("pk_source")
class TakesInConstructor(ChecksWiring):
    """Takes the prerequisite itself (not a module: it is rebuilt)."""

    def __init__(self, config=None):
        StatefulResource.__init__(self, config)
        self.get_dependency("pk_source")


@pytest.mark.parametrize("duplicate", [pickle.dumps, copy.copy, copy.deepcopy])
@pytest.mark.parametrize(
    "component_class", [TakesInConstructor, ChecksWiring, ChecksUndeclaredWiring],
)
def test_a_rebuilt_component_whose_constructor_consulted_its_wiring_refuses_to_travel(
        tmp_path, component_class, duplicate,
):
    _register_unpicklable_source()
    resource("pk_wiring", overwrite=True)(component_class)
    session = build_session(tmp_path, {"pk_source": {}, "pk_wiring": {}})
    asked = component_class.asked_about

    with pytest.raises(
            TypeError, match=rf"pk_wiring cannot be pickled or copied .*constructor.*{asked}",
    ):
        duplicate(resource_named(session, "pk_wiring"))


# -- a ModuleResource is pickled and copied as a module ------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_deep_copied_module_resource_stays_on_its_device():
    original = Encoder().to("cuda")

    copied = copy.deepcopy(original)

    assert copied.linear.weight.device.type == "cuda"


def test_a_deep_copied_module_resource_keeps_frozen_parameters_and_hooks():
    original = Encoder()
    original.linear.bias.requires_grad_(False)
    original.linear.register_forward_hook(lambda module, inputs, output: None)

    copied = copy.deepcopy(original)

    assert not copied.linear.bias.requires_grad
    assert len(copied.linear._forward_hooks) == 1


@pytest.mark.parametrize("duplicate", [_pickled, copy.deepcopy])
def test_weights_a_module_resource_shares_stay_shared(duplicate):
    container = nn.ModuleDict({"encoder": Encoder(), "output": nn.Linear(2, 2)})
    container["output"].weight = container["encoder"].linear.weight

    duplicated = duplicate(container)

    assert duplicated["output"].weight is duplicated["encoder"].linear.weight


def test_a_composed_module_resource_deep_copies_with_its_parts(tmp_path):
    resource("pk_encoder", overwrite=True)(Encoder)
    resource("pk_composed", overwrite=True)(ComposedAtConstruction)
    session = build_session(tmp_path, {"pk_encoder": {}, "pk_composed": {}})
    original = resource_named(session, "pk_composed")

    copied = copy.deepcopy(original)

    torch.testing.assert_close(
        copied.encoder.linear.weight, original.encoder.linear.weight,
    )
    assert copied.encoder is not original.encoder
    assert not copied.has_dependency("pk_encoder")


def test_a_module_resource_pickled_by_main_still_loads(monkeypatch):
    # What pickling a ModuleResource wrote up to main: the class, made by
    # __new__, then the version-1 envelope as its state.
    source = Encoder()
    with torch.no_grad():
        source.linear.weight.fill_(0.25)
    envelope = {
        "__training_framework_pickle_version__": 1,
        "init_args": {"args": (), "kwargs": {}},
        "state": source.get_state(),
    }
    monkeypatch.setattr(
        Encoder,
        "__reduce_ex__",
        lambda self, protocol: (copyreg.__newobj__, (Encoder,), envelope),
        raising=False,
    )
    earlier = pickle.dumps(Encoder.__new__(Encoder))
    monkeypatch.undo()

    restored = pickle.loads(earlier)

    torch.testing.assert_close(restored.linear.weight, source.linear.weight)
