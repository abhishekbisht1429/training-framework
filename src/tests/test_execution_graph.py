from __future__ import annotations

from training_framework.components import (
    LifecycleHook,
    Resource,
    Step,
    hook,
    requires_hook,
    requires_resource,
    resource,
    step,
)
from training_framework.session import TrainingSession


def _config(tmp_path, *, show_execution_graph=None):
    session_config = {
        "rng_seed": 1,
        "sessions_dir": str(tmp_path),
        "max_iterations": 3,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
    }
    if show_execution_graph is not None:
        session_config["show_execution_graph"] = show_execution_graph
    return {"session_config": session_config}


def _add_graph_components(session, setup_marker=None):
    @resource("graph_resource")
    class GraphResource(Resource):
        def setup(self, session):
            if setup_marker is not None:
                print(setup_marker)

        def teardown(self, session):
            pass

    @hook("graph_hook")
    @requires_resource("graph_resource")
    class GraphHook(LifecycleHook):
        call_every = 2

        def pre_session(self, session):
            pass

        def post_session(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @step("graph_step")
    @requires_hook("graph_hook")
    class GraphStep(Step):
        def run(self, session):
            pass

    @step("unused_graph_step")
    class UnusedGraphStep(Step):
        def run(self, session):
            pass

    session.register_resource(GraphResource())
    session.register_hook(GraphHook())
    session.add_step(GraphStep())


def test_execution_graph_expands_component_functions_in_runtime_order(tmp_path):
    session = TrainingSession(_config(tmp_path))
    _add_graph_components(session)

    graph = session.execution_graph()

    resource_setup = graph.index("Resource.graph_resource.setup()")
    hook_setup = graph.index("Hook.graph_hook.pre_session()")
    hook_pre = graph.index("Hook.graph_hook.pre_iteration_callback()")
    step_run = graph.index("Step.graph_step.run()")
    hook_post = graph.index("Hook.graph_hook.post_iteration_callback()")
    resource_teardown = graph.index("Resource.graph_resource.teardown()")
    hook_teardown = graph.index("Hook.graph_hook.post_session()")

    assert resource_setup < hook_setup < hook_pre < step_run < hook_post
    assert hook_post < hook_teardown < resource_teardown
    assert "requires: Resource.graph_resource" in graph
    assert "requires: Hook.graph_hook" in graph
    assert "cadence: first, every 2, final" in graph
    assert "Step.unused_graph_step.run()" not in graph


def test_execution_graph_prints_before_setup_by_default(tmp_path, capsys):
    session = TrainingSession(_config(tmp_path))
    _add_graph_components(session, setup_marker="GRAPH RESOURCE SETUP")

    with session:
        pass

    output = capsys.readouterr().out
    assert output.index("TRAINING SESSION EXECUTION GRAPH") < output.index(
        "GRAPH RESOURCE SETUP"
    )
    assert session.full_config["session_config"]["show_execution_graph"] is True


def test_execution_graph_automatic_print_can_be_disabled(tmp_path, capsys):
    session = TrainingSession(_config(tmp_path, show_execution_graph=False))

    with session:
        pass

    assert "TRAINING SESSION EXECUTION GRAPH" not in capsys.readouterr().out
    assert "TRAINING SESSION EXECUTION GRAPH" in session.execution_graph()
