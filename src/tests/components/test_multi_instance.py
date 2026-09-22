"""Tests for configuring a component more than once in one session.

This is the restriction the earlier phases were groundwork for: a top-level
`name#suffix` key activates its own instance, with its own configuration and
its own state.
"""

import pytest

from tests.test_utils import component_named, component_names, make_config
from training_framework.components import (
    ComponentDependencyError,
    Resource,
    requires_resource,
    resource,
    singleton,
)
from training_framework.session import TrainingSession


def declare_dependency(name="multi_dep"):
    @resource(name)
    class Dependency(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.tag = (config or {}).get("tag")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    return Dependency


def declare_consumer(dependency_name="multi_dep", name="multi_consumer"):
    @requires_resource(dependency_name)
    @resource(name)
    class Consumer(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.dependency = self.get_dependency(dependency_name)

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    return Consumer


# -- activating more than one instance ------------------------------------


def test_a_suffixed_key_activates_its_own_instance(tmp_path):
    declare_dependency()
    config = make_config(tmp_path / "two-instances")
    config["multi_dep"] = {"tag": "first"}
    config["multi_dep#b"] = {"tag": "second"}

    session = TrainingSession(config)
    first = component_named(session, "multi_dep")
    second = component_named(session, "multi_dep#b")

    assert first.tag == "first"
    assert second.tag == "second"
    assert first is not second


def test_every_instance_may_be_suffixed(tmp_path):
    declare_dependency()
    config = make_config(tmp_path / "all-suffixed")
    config["multi_dep#a"] = {"tag": "a"}
    config["multi_dep#b"] = {"tag": "b"}

    session = TrainingSession(config)

    assert "multi_dep" not in component_names(session)
    assert component_named(session, "multi_dep#a").tag == "a"
    assert component_named(session, "multi_dep#b").tag == "b"


def test_each_instance_keeps_its_own_state(tmp_path):
    declare_dependency()
    config = make_config(tmp_path / "separate-state")
    config["multi_dep#a"] = {"tag": "a"}
    config["multi_dep#b"] = {"tag": "b"}
    session = TrainingSession(config)

    state = session.get_state()
    restored = TrainingSession.from_state(state)

    assert sorted(state["components_state"])[:2] == ["checkpointer", "logger"]
    assert component_named(restored, "multi_dep#a").tag == "a"
    assert component_named(restored, "multi_dep#b").tag == "b"


def test_a_checkpointed_instance_records_its_component(tmp_path):
    declare_dependency()
    config = make_config(tmp_path / "state-implementation")
    config["multi_dep#b"] = {"tag": "b"}

    state = TrainingSession(config).get_state()

    entry = state["components_state"]["multi_dep#b"]
    assert entry["implementation"] == "multi_dep"


# -- what a dependency resolves to ----------------------------------------


def test_a_dependency_uses_the_only_instance(tmp_path):
    declare_dependency()
    declare_consumer()
    config = make_config(tmp_path / "sole-instance")
    config["multi_dep#b"] = {"tag": "b"}
    config["multi_consumer"] = {}

    session = TrainingSession(config)

    consumer = component_named(session, "multi_consumer")
    assert consumer.dependency is component_named(session, "multi_dep#b")


def test_a_dependency_prefers_the_unsuffixed_instance(tmp_path):
    declare_dependency()
    declare_consumer()
    config = make_config(tmp_path / "exact-wins")
    config["multi_dep"] = {"tag": "plain"}
    config["multi_dep#b"] = {"tag": "b"}
    config["multi_consumer"] = {}

    session = TrainingSession(config)

    # Adding an instance must not rewire a consumer that already worked.
    assert component_named(session, "multi_consumer").dependency.tag == "plain"


def test_an_undecidable_dependency_is_rejected(tmp_path):
    declare_dependency()
    declare_consumer()
    config = make_config(tmp_path / "ambiguous")
    config["multi_dep#a"] = {"tag": "a"}
    config["multi_dep#b"] = {"tag": "b"}
    config["multi_consumer"] = {}

    with pytest.raises(ComponentDependencyError) as error:
        TrainingSession(config)

    message = str(error.value)
    assert "multi_dep#a" in message and "multi_dep#b" in message
    assert "component_bindings" in message


def test_ambiguity_is_reported_whatever_the_configuration_order(tmp_path):
    declare_dependency()
    declare_consumer()
    config = make_config(tmp_path / "ambiguous-order")
    # The consumer comes first, so only one instance exists when it is
    # activated. Resolving against what happens to be built would pick it.
    config["multi_consumer"] = {}
    config["multi_dep#a"] = {"tag": "a"}
    config["multi_dep#b"] = {"tag": "b"}

    with pytest.raises(ComponentDependencyError):
        TrainingSession(config)


def test_a_consumer_may_name_the_instance_it_wants(tmp_path):
    declare_dependency()
    declare_consumer()
    config = make_config(tmp_path / "wired")
    config["component_bindings"] = {
        "multi_consumer": {"multi_dep": "multi_dep#b"},
    }
    config["multi_dep#a"] = {"tag": "a"}
    config["multi_dep#b"] = {"tag": "b"}
    config["multi_consumer"] = {}

    session = TrainingSession(config)

    consumer = component_named(session, "multi_consumer")
    assert consumer.dependency.tag == "b"
    assert consumer.linked_components == {"multi_dep": "multi_dep#b"}


# -- components that must stay unique -------------------------------------


def test_a_singleton_component_may_not_be_configured_twice(tmp_path):
    @singleton
    @resource("multi_only_one")
    class OnlyOne(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    config = make_config(tmp_path / "singleton")
    config["multi_only_one"] = {}
    config["multi_only_one#b"] = {}

    with pytest.raises(ValueError, match="only one instance"):
        TrainingSession(config)


def test_a_singleton_component_may_still_be_configured_once(tmp_path):
    @singleton
    @resource("multi_only_one")
    class OnlyOne(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    config = make_config(tmp_path / "singleton-once")
    config["multi_only_one#b"] = {}

    session = TrainingSession(config)

    assert "multi_only_one#b" in component_names(session)


def test_the_ddp_resource_is_a_singleton():
    from training_framework.components.builtin.distributed import DDPResource

    # init_process_group is process-wide, and the engine, worker and session
    # all fetch "ddp" expecting exactly one.
    assert DDPResource.singleton is True
