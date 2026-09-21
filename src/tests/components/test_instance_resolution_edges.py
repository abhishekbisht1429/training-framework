"""Edges of multi-instance resolution that failed silently or not at all.

Each of these once produced a session that worked and used the wrong
component, or refused an operation that should work:

- wiring to an instance that is not configured used a sibling instead, and
  with the sibling out of the way would have created the instance from
  nothing;
- a suffixed instance could not be removed;
- a stateful instance pickled on its own came back under its class's name;
- `@singleton` held only for the instances one activation call planned.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pickle

import pytest
from torch import nn

from tests.test_utils import make_config
from training_framework.components import (
    ComponentDependencyError,
    Resource,
    requires_resource,
    resource,
)
from training_framework.components.builtin.distributed import DDPResource
from training_framework.session import TrainingSession


def declare_dependency():
    @resource("edge_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    return Dependency


def declare_consumer(seen):
    @requires_resource("edge_role")
    @resource("edge_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            seen.append(self.get_dependency("edge_role").name)

        def teardown(self, session) -> None:
            pass

    return Consumer


def base_config(tmp_path):
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    return config


def resource_names(session):
    return sorted(component.name for component in session.get_all_resources())


# -- wiring to an instance that is not configured ---------------------------


def test_wiring_to_an_unconfigured_instance_is_an_error_not_a_sibling(
        tmp_path,
):
    declare_dependency()
    declare_consumer([])
    config = base_config(tmp_path)
    config["edge_dep#a"] = {}
    config["edge_consumer"] = {}
    config["component_bindings"] = {
        "edge_consumer": {"edge_role": "edge_dep#typo"},
    }

    with pytest.raises(
            ComponentDependencyError,
            match=r"'edge_dep#typo' is not configured.*\['edge_dep#a'\]",
    ):
        TrainingSession(config)


def test_an_unsuffixed_binding_still_finds_the_only_instance(tmp_path):
    declare_dependency()
    seen = []
    declare_consumer(seen)
    config = base_config(tmp_path)
    config["edge_dep#a"] = {}
    config["edge_consumer"] = {}
    config["component_bindings"] = {"edge_role": "edge_dep"}

    with TrainingSession(config):
        pass

    assert seen == ["edge_dep#a"]


# -- removing an instance ---------------------------------------------------


def test_a_suffixed_instance_can_be_removed(tmp_path):
    declare_dependency()
    config = base_config(tmp_path)
    config["edge_dep#a"] = {}
    config["edge_dep#b"] = {}
    session = TrainingSession(config)

    session.unregister_resource("edge_dep#b")

    assert "edge_dep#b" not in resource_names(session)
    assert "edge_dep#a" in resource_names(session)


# -- pickling one instance on its own ---------------------------------------


def test_a_stateful_instance_keeps_its_name_through_a_pickle(tmp_path):
    config = base_config(tmp_path)
    config["checkpointer#nightly"] = {"checkpoint_every": 10}
    session = TrainingSession(config)
    [nightly] = [
        hook for hook in session.get_all_hooks()
        if hook.name == "checkpointer#nightly"
    ]

    restored = pickle.loads(pickle.dumps(nightly))

    assert restored.name == "checkpointer#nightly"
    assert restored.id == "Hook.checkpointer#nightly"
    # What the name is for: the nightly checkpointer keeps its own directory
    # rather than writing on top of the plain one.
    assert restored.instance_suffix == "nightly"


def test_an_unnamed_stateful_component_still_pickles(tmp_path):
    config = base_config(tmp_path)
    session = TrainingSession(config)
    [plain] = [
        hook for hook in session.get_all_hooks() if hook.name == "checkpointer"
    ]

    restored = pickle.loads(pickle.dumps(plain))

    assert restored.name == "checkpointer"
    assert restored.instance_suffix is None


# -- @singleton at every entry point ----------------------------------------


def _ddp_config(tmp_path, ddp_key="ddp"):
    @resource("edge_model")
    class Model(nn.Linear, Resource):
        def __init__(self, config=None):
            nn.Linear.__init__(self, 1, 1)

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    config = base_config(tmp_path)
    config["component_bindings"] = {"model": "edge_model"}
    config["edge_model"] = {}
    config[ddp_key] = _ddp_settings()
    return config


def _ddp_settings():
    return {
        "world_size": 1,
        "backend": "gloo",
        "master_addr": "127.0.0.1",
        "master_port": "29518",
    }


def test_a_second_activation_cannot_add_a_second_singleton(tmp_path):
    session = TrainingSession(_ddp_config(tmp_path))

    with pytest.raises(ValueError, match="allows only one instance"):
        session.activate_component("ddp#second", _ddp_settings())

    assert [name for name in resource_names(session) if "ddp" in name] == [
        "ddp",
    ]


def test_hand_registration_cannot_add_a_second_singleton(tmp_path):
    session = TrainingSession(_ddp_config(tmp_path, ddp_key="ddp#only"))

    with pytest.raises(ValueError, match="allows only one instance"):
        session.register_resource(DDPResource(_ddp_settings()))


def test_a_checkpoint_holding_two_singletons_is_rejected(tmp_path):
    state = TrainingSession(_ddp_config(tmp_path)).get_state()
    state["components_state"]["ddp#forged"] = (
        state["components_state"]["ddp"]
    )

    with pytest.raises(ValueError, match="allows only one instance"):
        TrainingSession.from_state(state)


def test_a_rejected_restore_leaves_the_session_as_it_was(tmp_path):
    session = TrainingSession(_ddp_config(tmp_path))
    before = resource_names(session)
    state = session.get_state()
    state["components_state"]["ddp#forged"] = (
        state["components_state"]["ddp"]
    )

    with pytest.raises(ValueError, match="allows only one instance"):
        session.set_state(state)

    assert resource_names(session) == before
