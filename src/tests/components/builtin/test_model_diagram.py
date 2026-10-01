"""`model_diagram`: the model's data flow and its component wiring, drawn
into the session dir from the model's first forward call."""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field

import matplotlib
import pytest
import torch

from tests.test_utils import resource_named
from training_framework.components.builtin import Checkpointer
from training_framework.components.builtin.diagram import graph as graph_module
from training_framework.components.builtin.diagram import component as component_module
from training_framework.engine import load_session_for_worker
from training_framework.session import TrainingSession

D = 16
PATCH = "training_framework.components.builtin.transformer.ConvPatchEmbedding"


def model_config(tmp_path, *, diagram=None, layers=3, **extra):
    config = {
        "session_config": {
            "rng_seed": 3, "sessions_dir": str(tmp_path), "max_iterations": 1,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
            "show_execution_graph": False,
        },
        "role_bindings": {
            "model": "pooled_patch_transformer",
            "patch_embedding": "conv_patch_embedding",
            "positional_embedding": "learned_positional_embedding_2d",
            "sequence_encoder": "torch_transformer_encoder",
            "pooling": "attention_pooling",
            "pooling_query": "conditioned_pooling_query",
        },
        "pooled_patch_transformer": {"class_token": True},
        "conv_patch_embedding": {"in_channels": 3, "patch_size": 4, "embed_dim": D},
        "learned_positional_embedding_2d": {"grid_size": [2, 2], "embed_dim": D},
        "torch_transformer_encoder": {
            "embed_dim": D, "num_heads": 2, "num_layers": layers,
            "dim_feedforward": 16, "dropout": 0.0,
        },
        "attention_pooling": {"embed_dim": D, "num_heads": 2},
        "conditioned_pooling_query": {
            "embed_dim": D,
            "inputs": {
                "obj_patch": {
                    "module": PATCH, "in_channels": 3, "patch_size": 4,
                    "embed_dim": D, "reduce": "mean",
                },
                "obj_patch_location": {
                    "module": "torch.nn.Linear", "in_features": 2, "out_features": D,
                },
            },
            "hidden_dims": [D],
        },
        "model_diagram": {} if diagram is None else diagram,
    }
    config.update(extra)
    return config


def inputs():
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(2, 3, 8, 8, generator=generator),
        {
            "obj_patch": torch.randn(2, 3, 4, 4, generator=generator),
            "obj_patch_location": torch.randn(2, 2, generator=generator),
        },
    )


def run(session, calls=1):
    """Enter the session and call the model `calls` times; the outputs."""
    images, conditioning = inputs()
    outputs = []
    with session:
        model = resource_named(session, "model").eval()
        for _ in range(calls):
            outputs.append(model(images, **conditioning))
    return outputs


def read(session, name):
    path = os.path.join(session.session_config.session_dir, name)
    with open(path, encoding="utf-8") as file:
        return file.read()


def written(session):
    return sorted(
        name for name in os.listdir(session.session_config.session_dir)
        if name.startswith("model_diagram")
    )


def assert_png(path):
    with open(path, "rb") as file:
        assert file.read(8) == b"\x89PNG\r\n\x1a\n"


# -- reading a diagram back --------------------------------------------------------
#
# The tests read the written Graphviz file back into what it says -- nodes
# (their label lines), edges (label, style), clusters -- and assert on that,
# not on how the file spells it.

_TEXT = r'"((?:[^"\\]|\\.)*)"'
_NODE = re.compile(rf'^\s*(n\d+) \[label={_TEXT}')
_EDGE = re.compile(r"^\s*(n\d+) -> (n\d+)(?: \[(.*)\])?;$")
_CLUSTER = re.compile(rf"^\s*label={_TEXT}; style=")
_ATTRIBUTE = re.compile(rf"(\w+)=(?:{_TEXT}|(\w+))")


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m[1] == "n" else m[1], text)


@dataclass
class Diagram:
    nodes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    edges: list[tuple[str, str, str, str]] = field(default_factory=list)
    """(source id, target id, label, style)."""
    clusters: list[str] = field(default_factory=list)

    def named(self, first_line: str) -> list[tuple[str, ...]]:
        """The labels of the nodes whose first line is `first_line`."""
        return [lines for lines in self.nodes.values() if lines[0] == first_line]

    def node(self, first_line: str) -> tuple[str, ...]:
        (lines,) = self.named(first_line)
        return lines

    def edges_between(self, source: str, target: str) -> list[tuple[str, str]]:
        """(label, style) of every edge from a node named `source` to one
        named `target` (by first label line)."""
        return [
            (label, style) for s, t, label, style in self.edges
            if self.nodes[s][0] == source and self.nodes[t][0] == target
        ]

    def edges_from(self, source: str) -> list[tuple[str, str]]:
        """(target first line, label) of every edge leaving `source`."""
        return [
            (self.nodes[t][0], label) for s, t, label, _ in self.edges
            if self.nodes[s][0] == source
        ]


def read_diagram(path) -> Diagram:
    diagram = Diagram()
    with open(path, encoding="utf-8") as file:
        for line in file:
            if match := _NODE.match(line):
                diagram.nodes[match[1]] = tuple(_unescape(match[2]).split("\n"))
            elif match := _EDGE.match(line):
                attributes = {
                    key: bare or _unescape(quoted)
                    for key, quoted, bare in _ATTRIBUTE.findall(match[3] or "")
                }
                diagram.edges.append((
                    match[1], match[2], attributes.get("label", ""),
                    attributes.get("style", "solid"),
                ))
            elif match := _CLUSTER.match(line):
                diagram.clusters.append(_unescape(match[1]))
    return diagram


def diagram_of(session, stem="model_diagram") -> Diagram:
    return read_diagram(os.path.join(session.session_config.session_dir, f"{stem}.dot"))


# -- what is written -------------------------------------------------------------


def test_every_diagram_is_written_to_the_session_dir(tmp_path):
    session = TrainingSession(model_config(tmp_path))
    run(session)

    assert written(session) == [
        f"model_diagram{part}.{ext}"
        for part in ("", "_components", "_session")
        for ext in ("dot", "mmd", "png", "svg")
    ]
    directory = session.session_config.session_dir
    assert_png(os.path.join(directory, "model_diagram.png"))
    assert "<svg" in read(session, "model_diagram.svg")


def test_the_model_diagram_follows_the_data_with_shapes(tmp_path):
    session = TrainingSession(model_config(tmp_path))
    run(session)
    diagram = diagram_of(session)

    # Inputs by argument name, keyword inputs by their own key, with shapes.
    assert diagram.node("images") == ("images", "[2, 3, 8, 8]")
    assert diagram.node("obj_patch") == ("obj_patch", "[2, 3, 4, 4]")
    assert diagram.node("obj_patch_location") == ("obj_patch_location", "[2, 2]")
    assert diagram.node("output") == ("output", "[2, 1, 16]")
    # Framework blocks are clusters naming the role and the instance in it.
    assert any(
        "sequence_encoder" in cluster and "torch_transformer_encoder" in cluster
        for cluster in diagram.clusters
    )
    # A container module is named by its class.
    assert "ModuleDict" in diagram.node("encoders")
    # An operation between modules is a node, and edges carry shapes: the
    # two encoded inputs go into `cat`, which feeds the query projection.
    assert [label for _, label in diagram.edges_from("cat")] == ["[2, 32]"]
    into_cat = [label for s, t, label, _ in diagram.edges if diagram.nodes[t][0] == "cat"]
    assert into_cat and set(into_cat) == {"[2, 16]"}
    # Shape-only operations are folded into the edges by default.
    assert not diagram.named("transpose") and not diagram.named("flatten")
    mermaid = read(session, "model_diagram.mmd")
    assert "flowchart TD" in mermaid and "obj_patch_location" in mermaid


def test_shape_operations_can_be_shown(tmp_path):
    session = TrainingSession(model_config(tmp_path, diagram={"show_shape_ops": True}))
    run(session)

    assert diagram_of(session).named("transpose")


def test_depth_shows_deeper_modules_and_collapses_repeated_layers(tmp_path):
    session = TrainingSession(model_config(tmp_path, diagram={"depth": 4}))
    run(session)
    diagram = diagram_of(session)

    # The three encoder layers are one node, saying there are three.
    (layers,) = [lines for lines in diagram.nodes.values()
                 if any("TransformerEncoderLayer" in line for line in lines)]
    assert "x3" in " ".join(layers)
    # Deeper modules appear: the query's encoder for `obj_patch_location`,
    # beside the input of that name.
    assert sorted(diagram.named("obj_patch_location")) == [
        ("obj_patch_location", "Linear"), ("obj_patch_location", "[2, 2]"),
    ]


def test_repeated_layers_can_be_drawn_one_by_one(tmp_path):
    session = TrainingSession(model_config(
        tmp_path, diagram={"depth": 4, "collapse_repeats": False},
    ))
    run(session)
    diagram = diagram_of(session)

    layers = [lines for lines in diagram.nodes.values()
              if "TransformerEncoderLayer" in lines[1:]]
    assert sorted(lines[0] for lines in layers) == ["0", "1", "2"]


def test_the_component_wiring_names_roles_and_parameters(tmp_path):
    session = TrainingSession(model_config(tmp_path))
    for parameter in resource_named(session, "conv_patch_embedding").parameters():
        parameter.requires_grad_(False)
    run(session)
    diagram = diagram_of(session, "model_diagram_components")

    # Each role, from the model to the instance that fills it.
    assert sorted(diagram.edges_from("pooled_patch_transformer")) == [
        ("attention_pooling", "pooling"),
        ("conditioned_pooling_query", "pooling_query"),
        ("conv_patch_embedding", "patch_embedding"),
        ("learned_positional_embedding_2d", "positional_embedding"),
        ("torch_transformer_encoder", "sequence_encoder"),
    ]
    # Class, and the parameters a component holds itself (the model's own
    # 16 are its class token; its blocks count theirs), frozen apart.
    model = " ".join(diagram.node("pooled_patch_transformer"))
    assert "PooledPatchTransformer" in model and "16" in model
    patch = " ".join(diagram.node("conv_patch_embedding"))
    assert "784" in patch and "frozen" in patch


def test_the_component_wiring_can_be_left_out(tmp_path):
    session = TrainingSession(model_config(tmp_path, diagram={"component_wiring": False}))
    run(session)

    assert written(session) == [f"model_diagram.{ext}" for ext in ("dot", "mmd", "png", "svg")]


@pytest.mark.parametrize("scope, drawn", [
    ("model", ["_components"]),
    ("session", ["_session"]),
    ("both", ["_components", "_session"]),
])
def test_scope_picks_the_component_wiring_drawn(tmp_path, scope, drawn):
    session = TrainingSession(model_config(tmp_path, diagram={"scope": scope}))
    run(session)

    wiring = sorted({
        name.split(".")[0][len("model_diagram"):]
        for name in written(session)
    } - {""})
    assert wiring == drawn


def test_the_session_wiring_holds_every_component_by_kind(tmp_path):
    session = TrainingSession(model_config(tmp_path, diagram={"scope": "session"}))
    run(session)
    diagram = diagram_of(session, "model_diagram_session")

    # This session has no steps, and an empty kind gets no box.
    assert {"Resources", "Hooks"} <= set(diagram.clusters)
    assert "Steps" not in diagram.clusters
    for name in ("pooled_patch_transformer", "attention_pooling", "logger",
                 "checkpointer", "model_diagram"):
        assert diagram.named(name)
    # A requirement is labelled with the role when the instance differs.
    assert diagram.edges_between("model_diagram", "pooled_patch_transformer") == [
        ("model", "solid"),
    ]
    mermaid = read(session, "model_diagram_session.mmd")
    assert 'subgraph' in mermaid and '"Hooks"' in mermaid


def test_an_imported_component_says_where_it_came_from(tmp_path):
    source = TrainingSession(model_config(tmp_path / "source", diagram={"component_wiring": False}))
    saved = Checkpointer.save_checkpoint(source, tmp_path / "saved")
    config = model_config(tmp_path / "run")
    config["import_components"] = {"pooled_patch_transformer": {"checkpoint": str(saved)}}
    config["role_bindings"] = {"model": "pooled_patch_transformer#imported"}
    for name in list(config):
        if name not in ("session_config", "role_bindings", "import_components", "model_diagram"):
            del config[name]
    session = TrainingSession(config)
    run(session)

    diagram = diagram_of(session, "model_diagram_components")
    assert "imported by import_components.pooled_patch_transformer" in (
        diagram.node("pooled_patch_transformer#imported")
    )


def test_a_suffixed_instance_writes_its_own_files(tmp_path):
    config = model_config(tmp_path)
    config["model_diagram#b"] = {"component_wiring": False}
    session = TrainingSession(config)
    run(session)

    assert "model_diagram_b.png" in written(session)
    assert "model_diagram.png" in written(session)


# -- the run is not changed ----------------------------------------------------------


def test_the_forward_call_is_unchanged_and_recorded_once(tmp_path):
    session = TrainingSession(model_config(tmp_path))
    recorded, later = run(session, calls=2)

    torch.testing.assert_close(recorded, later)
    # The diagram is the first call's.
    assert diagram_of(session).node("images") == ("images", "[2, 3, 8, 8]")
    model = resource_named(session, "model")
    # Every hook the recording installed is gone after the first call.
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in model.modules()
    )


def test_only_the_first_call_is_drawn(tmp_path):
    session = TrainingSession(model_config(tmp_path, diagram={"component_wiring": False}))
    images, conditioning = inputs()
    with session:
        model = resource_named(session, "model").eval()
        model(images, **conditioning)
        # A later call with another batch size changes nothing drawn.
        model(images[:1], **{key: value[:1] for key, value in conditioning.items()})

    assert diagram_of(session).node("images") == ("images", "[2, 3, 8, 8]")


def test_the_matplotlib_backend_is_left_alone(tmp_path):
    backend = matplotlib.get_backend()
    session = TrainingSession(model_config(tmp_path))
    run(session)

    assert matplotlib.get_backend() == backend


def test_without_graphviz_matplotlib_draws_with_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_module.shutil, "which", lambda name: None)
    session = TrainingSession(model_config(tmp_path))

    with pytest.warns(RuntimeWarning, match=r"Graphviz \(`dot`\) is not on PATH; model_diagram was drawn with matplotlib instead\. Install Graphviz"):
        run(session)

    assert_png(os.path.join(session.session_config.session_dir, "model_diagram.png"))
    assert diagram_of(session).named("cat")


def test_a_drawing_failure_warns_and_the_run_goes_on(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no room")

    monkeypatch.setattr(component_module, "build_model_graph", broken)
    session = TrainingSession(model_config(tmp_path, diagram={"component_wiring": False}))

    with pytest.warns(RuntimeWarning, match=r"model_diagram: drawing model_diagram failed: RuntimeError\('no room'\)"):
        recorded, later = run(session, calls=2)
    torch.testing.assert_close(recorded, later)


def test_a_model_that_never_runs_is_reported(tmp_path):
    session = TrainingSession(model_config(tmp_path, diagram={"component_wiring": False}))

    with pytest.warns(RuntimeWarning, match="the model never ran a forward call"):
        with session:
            pass
    assert written(session) == []


def test_only_rank_zero_draws(tmp_path):
    config = model_config(tmp_path)
    config["ddp"] = {
        "world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1",
        "master_port": "12399",
    }
    parent = TrainingSession(config)

    rank_one = load_session_for_worker(parent.get_state(), rank=1)

    assert "model_diagram" in [hook.name for hook in parent.get_all_hooks()]
    assert "model_diagram" not in [hook.name for hook in rank_one.get_all_hooks()]


@pytest.mark.parametrize("diagram, message", [
    ({"depth": 0}, "depth must be a positive integer"),
    ({"formats": ["jpg"]}, r"formats must be a non-empty list of \['png', 'svg', 'pdf'\]"),
    ({"formats": []}, "formats must be a non-empty list"),
    ({"file": "../x"}, "file must be a file name without a directory"),
    ({"show_shape_ops": "yes"}, "show_shape_ops must be true or false"),
    ({"scope": "all"}, r"scope must be one of \['model', 'session', 'both'\]"),
])
def test_a_bad_configuration_is_refused_when_the_session_is_built(tmp_path, diagram, message):
    with pytest.raises(ValueError, match=message):
        TrainingSession(model_config(tmp_path, diagram=diagram))
