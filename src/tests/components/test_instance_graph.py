"""Tests that the component graph has one node per instance, not per class.

Configuration cannot yet ask for a second instance, so these tests construct
one through the session's own machinery and put it alongside the first. That
is exactly the state a later phase will reach from config, and it is what the
sort has to cope with: edges resolved to the registered class would collapse
every instance of a component onto a single node.
"""

import pytest

from tests.test_utils import make_config
from training_framework.components import (
    IterationHook,
    Resource,
    hook,
    requires_resource,
    resource,
    wraps,
)
from training_framework.components.graph import topological_sort_components
from training_framework.session import TrainingSession


def make_session_with_two_instances(tmp_path, name):
    """Return a session holding `graph_dep` twice, under two names."""

    @resource("graph_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("graph_dep")
    @resource("graph_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    config = make_config(tmp_path / name)
    config["graph_consumer"] = {}
    session = TrainingSession(config)

    components = session._components
    second = components._construct(Dependency, "graph_dep#2")
    components.components["graph_dep#2"] = second
    return session, second


def test_each_instance_is_its_own_node(tmp_path):
    session, second = make_session_with_two_instances(tmp_path, "two-nodes")

    order = session._components._component_order()

    assert "Resource.graph_dep" in order
    assert "Resource.graph_dep#2" in order
    assert order["Resource.graph_dep"] != order["Resource.graph_dep#2"]


def test_both_instances_are_ordered(tmp_path):
    session, second = make_session_with_two_instances(tmp_path, "two-ordered")

    ordered = session._components.ordered_components

    assert second in ordered
    names = [component.name for component in ordered]
    assert names.count("graph_dep") == 1
    assert names.count("graph_dep#2") == 1


def test_a_consumer_is_ordered_after_the_instance_it_names(tmp_path):
    session, _ = make_session_with_two_instances(tmp_path, "edge-target")

    order = session._components._component_order()

    # The dependency is declared as 'graph_dep', so that is the instance the
    # edge points at; the second instance is an unrelated node.
    assert order["Resource.graph_dep"] < order["Resource.graph_consumer"]


def test_a_second_instance_carries_no_edges_of_its_own(tmp_path):
    session, _ = make_session_with_two_instances(tmp_path, "no-edges")

    order = session._components._component_order()

    # Nothing depends on it, so it sorts among the roots rather than being
    # forced after the consumer that names the other instance.
    assert order["Resource.graph_dep#2"] < order["Resource.graph_consumer"]


def test_two_hook_instances_wrapping_one_target_are_distinct(tmp_path):
    @hook("graph_wrapped")
    class Wrapped(IterationHook):
        call_every = 1

        def pre_iteration_callback(self, session) -> None:
            pass

        def post_iteration_callback(self, session) -> None:
            pass

    @wraps("graph_wrapped")
    @hook("graph_wrapper")
    class Wrapper(IterationHook):
        call_every = 1

        def pre_iteration_callback(self, session) -> None:
            pass

        def post_iteration_callback(self, session) -> None:
            pass

    config = make_config(tmp_path / "wrapping-instances")
    config["graph_wrapper"] = {}
    session = TrainingSession(config)

    components = session._components
    components.components["graph_wrapper#2"] = components._construct(
        Wrapper,
        "graph_wrapper#2",
    )

    order = components._component_order()

    assert order["Hook.graph_wrapper"] != order["Hook.graph_wrapper#2"]
    # A wrapper runs before the hook it wraps, and both instances wrap it.
    assert order["Hook.graph_wrapper"] < order["Hook.graph_wrapped"]
    assert order["Hook.graph_wrapper#2"] < order["Hook.graph_wrapped"]


class InstanceResolver:
    """A binding resolver that points a role at one specific instance.

    `ComponentBindings` does not accept an instance name as a target yet --
    that is the next phase -- so the graph is exercised directly here through
    the resolver interface it actually depends on.
    """

    def __init__(self, bindings):
        self._bindings = dict(bindings)

    def resolve(self, name, *, consumer=None):
        return self._bindings.get(name, name)

    @property
    def bindings(self):
        return dict(self._bindings)

    def __bool__(self):
        return bool(self._bindings)


def test_an_edge_can_point_at_a_named_instance(tmp_path):
    session, second = make_session_with_two_instances(tmp_path, "edge-instance")
    components = session._components

    order = topological_sort_components(
        binding_resolver=InstanceResolver({"graph_dep": "graph_dep#2"}),
        registry=components.registry,
        components=list(components.components.values()),
        roles=components.roles,
        session_type=components.session_type,
    )

    # The consumer declares 'graph_dep', which now resolves to the second
    # instance, so that is the node it must be ordered after. Resolving the
    # edge to the registered class instead would point it at the first.
    assert order["Resource.graph_dep#2"] < order["Resource.graph_consumer"]


def test_an_edge_to_a_missing_instance_is_reported(tmp_path):
    session, _ = make_session_with_two_instances(tmp_path, "edge-missing")
    components = session._components

    with pytest.raises(RuntimeError, match="not configured in this session"):
        topological_sort_components(
            binding_resolver=InstanceResolver({"graph_dep": "graph_dep#9"}),
            registry=components.registry,
            components=list(components.components.values()),
            roles=components.roles,
            session_type=components.session_type,
        )


def test_an_unconfigured_prerequisite_is_still_reported(tmp_path):
    @resource("graph_absent_dep")
    class Absent(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("graph_absent_dep")
    @resource("graph_needy")
    class Needy(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    config = make_config(tmp_path / "absent-prerequisite")
    config["graph_needy"] = {}
    session = TrainingSession(config)

    # Remove the prerequisite the session activated, leaving the edge dangling.
    del session._components.components["graph_absent_dep"]

    with pytest.raises(RuntimeError, match="not configured in this session"):
        session._components._component_order()
