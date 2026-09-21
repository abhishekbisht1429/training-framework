"""The execution graph shows the wiring a component is actually given.

The graph is where a user checks how a session is wired, so it has to agree
with resolution. It used to resolve each `requires:` annotation without the
consumer -- showing a component wired to `graph_dep#b` as requiring whatever
the session-wide binding said -- and it printed no per-consumer wiring at
all.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_utils import make_config
from training_framework.components import Resource, requires_resource, resource
from training_framework.components.builtin.model import TrainedModel
from training_framework.session import TrainingSession


def declare_components():
    @resource("graph_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("graph_role")
    @resource("graph_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass


def wired_session(tmp_path, *, flat_binding):
    declare_components()
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["graph_dep#a"] = {}
    config["graph_dep#b"] = {}
    config["graph_consumer"] = {}
    bindings = {"graph_consumer": {"graph_role": "graph_dep#b"}}
    if flat_binding:
        bindings["graph_role"] = "graph_dep#a"
    config["component_bindings"] = bindings
    return TrainingSession(config)


def consumer_lines(graph):
    return [line for line in graph.splitlines() if "graph_consumer." in line]


def test_requires_names_the_instance_the_consumer_is_wired_to(tmp_path):
    graph = wired_session(tmp_path, flat_binding=True).execution_graph()

    lines = consumer_lines(graph)
    assert lines
    for line in lines:
        assert "requires: Resource.graph_dep#b" in line
        assert "graph_dep#a" not in line


def test_per_consumer_wiring_is_listed_with_the_bindings(tmp_path):
    graph = wired_session(tmp_path, flat_binding=True).execution_graph()

    assert "  graph_role -> graph_dep#a" in graph
    assert "  graph_consumer: graph_role -> graph_dep#b" in graph


def test_a_session_wired_only_per_consumer_still_shows_its_bindings(tmp_path):
    graph = wired_session(tmp_path, flat_binding=False).execution_graph()

    assert "COMPONENT BINDINGS" in graph
    assert "  graph_consumer: graph_role -> graph_dep#b" in graph


# -- a checkpoint that names no single model --------------------------------


class GraphTwinModel(nn.Linear, Resource):
    """Declared at module scope so the checkpoint can be unpickled."""

    def __init__(self, config=None):
        nn.Linear.__init__(self, 1, 1)
        self._config = config or {}

    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass


def test_trained_model_reports_a_checkpoint_with_two_candidate_models(
        tmp_path,
):
    resource("graph_twin_model")(GraphTwinModel)
    config = make_config(tmp_path / "source")
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"model": "graph_twin_model"}
    config["graph_twin_model#a"] = {}
    config["graph_twin_model#b"] = {}
    checkpoint = tmp_path / "two-models.pt"
    torch.save(TrainingSession(config), checkpoint)

    trained_model = TrainedModel({"model_checkpoint_path": str(checkpoint)})

    with pytest.raises(
            ValueError,
            match="does not contain a resolvable 'model' resource",
    ):
        trained_model.setup(SimpleNamespace(device=torch.device("cpu")))
