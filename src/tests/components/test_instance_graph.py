"""Tests that the component graph has one node per instance, not per class.

Edges resolved to the registered class would collapse every instance of a
component onto a single node. What that breaks is visible from outside: the
graph decides the order components are set up and called in, so each test
records that order.

The sort takes the nodes with nothing left to wait for in the order they were
activated, so every node without prerequisites comes before every node with
one. A test about where an edge points has to give the candidates different
depths, or it passes whichever way the edge goes.
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
from training_framework.session import TrainingSession


class TracedResource(Resource):
    """Records its instance name when it is set up."""

    trace: list[str]

    def setup(self, session) -> None:
        self.trace.append(self.name)

    def teardown(self, session) -> None:
        pass


def traced(name, trace, *, requires=None):
    """Register a resource `name` that records its setup in `trace`."""
    component = resource(name)(
        type(f"Traced_{name}", (TracedResource,), {"trace": trace})
    )
    if requires is not None:
        requires_resource(requires)(component)
    return component


def run(tmp_path, **components):
    config = make_config(tmp_path, max_iterations=1)
    config["session_config"]["show_execution_graph"] = False
    config.update(components)
    with TrainingSession(config) as session:
        list(session)


def two_instances_and_a_consumer(tmp_path):
    trace = []
    # `graph_dep` waits for `graph_base`, so it is not a root: an edge from
    # the consumer is the only thing ordering the consumer after it.
    traced("graph_base", trace)
    traced("graph_dep", trace, requires="graph_base")
    traced("graph_consumer", trace, requires="graph_dep")
    run(
        tmp_path,
        graph_base={},
        graph_dep={},
        **{"graph_dep#2": {}},
        graph_consumer={},
    )
    return trace


def test_each_instance_is_set_up_once(tmp_path):
    trace = two_instances_and_a_consumer(tmp_path)

    assert trace.count("graph_dep") == 1
    assert trace.count("graph_dep#2") == 1


def test_a_consumer_is_set_up_after_the_instance_it_names(tmp_path):
    trace = two_instances_and_a_consumer(tmp_path)

    # The dependency is declared as 'graph_dep', so that is the instance the
    # edge points at; the second instance is an unrelated node.
    assert trace.index("graph_dep") < trace.index("graph_consumer")


def test_a_second_instance_carries_no_edges_of_its_own(tmp_path):
    trace = two_instances_and_a_consumer(tmp_path)

    # Nothing depends on it, so it sorts among the roots rather than being
    # forced after the consumer that names the other instance.
    assert trace.index("graph_dep#2") < trace.index("graph_consumer")


def test_an_edge_can_point_at_a_named_instance(tmp_path):
    trace = []
    traced("graph_base", trace)
    traced("graph_mid", trace, requires="graph_base")
    traced("graph_dep", trace, requires="graph_source")
    traced("graph_consumer", trace, requires="graph_dep")

    run(
        tmp_path,
        component_bindings={
            # Two instances of one class, one level apart.
            "graph_dep": {"graph_source": "graph_base"},
            "graph_dep#2": {"graph_source": "graph_mid"},
            "graph_consumer": {"graph_dep": "graph_dep#2"},
        },
        graph_base={},
        graph_dep={},
        graph_mid={},
        **{"graph_dep#2": {}},
        graph_consumer={},
    )

    # Pointed at the shallower 'graph_dep' instead, the edge would let the
    # consumer be set up before the instance it was actually handed.
    assert trace.index("graph_dep#2") < trace.index("graph_consumer")


def test_two_hook_instances_wrapping_one_target_both_run_around_it(tmp_path):
    calls = []

    class TracedHook(IterationHook):
        call_every = 1

        def pre_iteration_callback(self, session) -> None:
            calls.append(("pre", self.name))

        def post_iteration_callback(self, session) -> None:
            calls.append(("post", self.name))

    traced("graph_base", [])
    traced("graph_mid", [], requires="graph_base")
    traced("graph_deep", [], requires="graph_mid")
    hook("graph_wrapped")(type("Wrapped", (TracedHook,), {}))
    requires_resource("graph_wrapper_source")(wraps("graph_wrapped")(
        hook("graph_wrapper")(type("Wrapper", (TracedHook,), {}))
    ))

    run(
        tmp_path,
        component_bindings={
            # The second instance sits deeper than the hook it wraps, so only
            # its own wrapping edge can order it first.
            "graph_wrapper": {"graph_wrapper_source": "graph_base"},
            "graph_wrapper#2": {"graph_wrapper_source": "graph_deep"},
        },
        graph_base={},
        graph_mid={},
        graph_deep={},
        graph_wrapper={},
        **{"graph_wrapper#2": {}},
    )

    pre = [name for phase, name in calls if phase == "pre"]
    post = [name for phase, name in calls if phase == "post"]
    # A wrapper runs before the hook it wraps and after it on the way out,
    # and both instances wrap it.
    assert pre.index("graph_wrapper") < pre.index("graph_wrapped")
    assert pre.index("graph_wrapper#2") < pre.index("graph_wrapped")
    assert post.index("graph_wrapped") < post.index("graph_wrapper")
    assert post.index("graph_wrapped") < post.index("graph_wrapper#2")


def test_an_edge_to_a_removed_instance_is_reported_not_given_a_sibling(
        tmp_path,
):
    trace = []
    traced("graph_dep", trace)
    traced("graph_consumer", trace, requires="graph_dep")
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"graph_consumer": {"graph_dep": "graph_dep#2"}}
    config.update({"graph_dep": {}, "graph_dep#2": {}, "graph_consumer": {}})
    session = TrainingSession(config)

    # The consumer has not taken it yet, so it may be removed; the edge then
    # names an instance that is not there, and must not fall back to its
    # sibling 'graph_dep'.
    session.unregister_resource("graph_dep#2")

    with pytest.raises(RuntimeError, match="not configured in this session"):
        with session:
            pass


def test_an_unconfigured_prerequisite_is_still_reported(tmp_path):
    trace = []
    traced("graph_absent_dep", trace)
    traced("graph_needy", trace, requires="graph_absent_dep")
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["graph_needy"] = {}
    session = TrainingSession(config)

    # Remove the prerequisite the session activated, leaving the edge dangling.
    session.unregister_resource("graph_absent_dep")

    with pytest.raises(RuntimeError, match="not configured in this session"):
        with session:
            pass
