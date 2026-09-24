"""Checkpoint directories: what is written, and reading parts of one.

A checkpoint is a directory of plain data -- a manifest, the session's own
state, one file per component -- so one component can be read or rebuilt
without the rest, and a class change breaks only what it touches.
"""

import json
import os
import shutil

import pytest
import torch

from tests.test_utils import make_config, resource_named
from training_framework.components import (
    Resource,
    Stateful,
    requires_resource,
    resource,
)
from training_framework.components.builtin import Checkpointer
from training_framework.session import TrainingSession

REBUILD_FAILS = {"fragile": False}
CONSTRUCTOR_NEEDS_SCALE = {"scaled": False}


class Weights(Resource, Stateful):
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.value = torch.full((2,), float(self.config.get("value", 0.0)))

    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass

    def get_state(self):
        return {"value": self.value.clone()}

    def set_state(self, state) -> None:
        self.value = state["value"].clone()


class Backbone(Weights):
    pass


@requires_resource("backbone")
class Head(Weights):
    pass


class Fragile(Weights):
    """Stands for a component whose class changed so it no longer builds."""

    def __init__(self, config=None):
        if REBUILD_FAILS["fragile"]:
            raise RuntimeError("fragile can no longer be built")
        super().__init__(config)


class Scaled(Weights):
    """Stands for a component whose constructor gained a required argument."""

    def __init__(self, config=None, *, scale=None):
        if CONSTRUCTOR_NEEDS_SCALE["scaled"] and scale is None:
            raise TypeError("Scaled() needs a scale")
        super().__init__(config)
        self.scale = scale


class Opaque(Weights):
    def get_state(self):
        return {"value": self.value.clone(), "callback": object()}


def build(tmp_path, bindings=None, **components):
    for name, component_class in (
            ("backbone", Backbone),
            ("head", Head),
            ("fragile", Fragile),
            ("opaque", Opaque),
            ("scaled", Scaled),
    ):
        resource(name)(component_class)
    config = make_config(tmp_path / "source")
    config["session_config"]["show_execution_graph"] = False
    if bindings is not None:
        config["component_bindings"] = bindings
    config.update(components)
    return TrainingSession(config)


def save(tmp_path, session=None, **components):
    session = session or build(tmp_path, **components)
    return Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")


def value_of(component) -> float:
    return component.value[0].item()


# -- the format --------------------------------------------------------------------


def test_a_checkpoint_is_a_directory_of_plain_data(tmp_path):
    path = save(tmp_path, backbone={"value": 1.0}, head={"value": 2.0})

    manifest = Checkpointer.read_manifest(path)
    with open(f"{path}/manifest.json") as manifest_file:
        assert manifest == json.load(manifest_file)
    assert manifest["session_type"] == "training"
    head = manifest["components"]["head"]
    assert head["implementation"] == "head"
    assert head["component_type"] == "Resource"
    assert head["dependencies"] == {"backbone": "backbone"}
    assert head["state_version"] == 1

    # Every file reads without importing a class.
    torch.load(f"{path}/session.pt", weights_only=True)
    for record in manifest["components"].values():
        torch.load(f"{path}/{record['file']}", weights_only=True)


def test_a_session_round_trips_through_a_checkpoint_directory(tmp_path):
    session = build(tmp_path, backbone={"value": 1.0}, head={"value": 2.0})
    resource_named(session, "backbone").value = torch.tensor([7.0, 7.0])
    path = save(tmp_path, session)

    restored = Checkpointer.load_checkpoint(path)

    assert restored.iteration == session.iteration
    assert value_of(resource_named(restored, "backbone")) == 7.0
    head = resource_named(restored, "head")
    assert head.get_dependency("backbone") is resource_named(restored, "backbone")


def test_state_that_is_not_plain_data_is_refused_when_saving(tmp_path):
    session = build(tmp_path, opaque={})

    with pytest.raises(ValueError, match=r"component 'opaque' state.*'callback'"):
        Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")

    assert list(tmp_path.glob("checkpoint*")) == []


def test_an_unfinished_checkpoint_is_not_read(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="not a checkpoint"):
        Checkpointer.load_checkpoint(tmp_path / "empty")

    path = save(tmp_path, backbone={})
    interrupted = tmp_path / "interrupted.tmp"
    (tmp_path / "checkpoint").rename(interrupted)
    with pytest.raises(ValueError, match="unfinished"):
        Checkpointer.load_checkpoint(interrupted)
    assert path


def test_a_damaged_component_file_is_reported_by_name(tmp_path):
    path = save(tmp_path, backbone={"value": 1.0})
    file = tmp_path / "checkpoint" / "components" / "backbone.pt"
    file.write_bytes(file.read_bytes()[:-8] + b"damaged!")

    with pytest.raises(ValueError, match="'backbone'.*checksum"):
        Checkpointer.load_checkpoint(path)


def test_a_single_file_checkpoint_is_still_read_with_a_warning(tmp_path):
    session = build(tmp_path, backbone={"value": 3.0})
    torch.save(session, tmp_path / "legacy.pt")

    with pytest.warns(FutureWarning, match="single-file checkpoint"):
        restored = Checkpointer.load_checkpoint(tmp_path / "legacy.pt")

    assert value_of(resource_named(restored, "backbone")) == 3.0


# -- reading one component -----------------------------------------------------------


def test_one_component_loads_when_another_no_longer_builds(
        tmp_path, monkeypatch,
):
    path = save(
        tmp_path,
        backbone={"value": 1.0},
        head={"value": 2.0},
        fragile={},
    )
    monkeypatch.setitem(REBUILD_FAILS, "fragile", True)

    with pytest.raises(RuntimeError, match="no longer be built"):
        Checkpointer.load_checkpoint(path)

    head = Checkpointer.load_component(path, "head")
    assert value_of(head) == 2.0
    assert value_of(head.get_dependency("backbone")) == 1.0
    assert value_of(Checkpointer.load_component(path, "backbone")) == 1.0


def test_a_component_wired_to_others_needs_its_dependencies(tmp_path):
    path = save(tmp_path, backbone={"value": 1.0}, head={"value": 2.0})

    with pytest.raises(ValueError, match="load_component_state"):
        Checkpointer.load_component(path, "head", with_dependencies=False)

    backbone = Checkpointer.load_component(
        path, "backbone", with_dependencies=False,
    )
    assert value_of(backbone) == 1.0


def test_a_saved_state_reads_without_building_anything(tmp_path, monkeypatch):
    path = save(
        tmp_path,
        bindings={"trunk": "backbone"},
        backbone={"value": 4.0},
        fragile={"value": 5.0},
    )
    monkeypatch.setitem(REBUILD_FAILS, "fragile", True)

    assert Checkpointer.load_component_state(path, "fragile")["value"][0] == 5.0
    assert Checkpointer.load_component_state(path, "trunk")["value"][0] == 4.0
    with pytest.raises(KeyError):
        Checkpointer.load_component_state(path, "no_such_component")


def test_one_component_is_rebuilt_with_the_instance_it_was_given(tmp_path):
    path = save(
        tmp_path,
        bindings={"head": {"backbone": "backbone#b"}},
        **{
            "backbone#a": {"value": 1.0},
            "backbone#b": {"value": 2.0},
            "head": {},
        },
    )

    manifest = Checkpointer.read_manifest(path)
    assert manifest["components"]["head"]["dependencies"] == {
        "backbone": "backbone#b",
    }
    head = Checkpointer.load_component(path, "head")
    assert head.get_dependency("backbone").name == "backbone#b"
    assert value_of(head.get_dependency("backbone")) == 2.0
    # The name head asked for resolves to what head was given.
    assert Checkpointer.load_component_state(path, "backbone")["value"][0] == 2.0


def test_the_checkpoint_must_hold_the_session_type_asked_for(tmp_path):
    path = save(tmp_path, backbone={})

    with pytest.raises(ValueError, match="must contain an analysis session"):
        Checkpointer.load_component(path, "backbone", session_type="analysis")


# -- state versions ------------------------------------------------------------------


def save_changed(tmp_path, **components):
    """Save a session whose backbone state differs from its configuration,
    so a restored state and a freshly built one can be told apart."""
    session = build(tmp_path, **components)
    resource_named(session, "backbone").value = torch.tensor([9.0, 9.0])
    return save(tmp_path, session)


def test_an_older_state_is_migrated(tmp_path, monkeypatch):
    path = save_changed(tmp_path, backbone={"value": 1.0})
    monkeypatch.setattr(Backbone, "state_version", 2)
    monkeypatch.setattr(
        Backbone,
        "migrate_state",
        classmethod(lambda cls, version, state: {"value": state["value"] * 10}),
    )

    restored = Checkpointer.load_checkpoint(path)

    assert value_of(resource_named(restored, "backbone")) == 90.0


def test_changed_constructor_arguments_are_migrated(tmp_path, monkeypatch):
    path = save(tmp_path, backbone={"value": 1.0})
    monkeypatch.setattr(Backbone, "state_version", 2)
    monkeypatch.setattr(
        Backbone,
        "migrate_init_args",
        classmethod(lambda cls, version, init_args: {
            "args": ({"value": 1.0, "renamed": True},),
            "kwargs": {},
        }),
    )
    monkeypatch.setattr(
        Backbone,
        "migrate_state",
        classmethod(lambda cls, version, state: state),
    )

    restored = Checkpointer.load_checkpoint(path)

    assert resource_named(restored, "backbone").config["renamed"] is True


def test_an_older_state_without_a_migration_is_refused(tmp_path, monkeypatch):
    path = save_changed(tmp_path, backbone={"value": 1.0})
    monkeypatch.setattr(Backbone, "state_version", 2)

    with pytest.raises(ValueError, match="'backbone'.*no migrate_state"):
        Checkpointer.load_checkpoint(path)


def test_a_newer_state_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(Backbone, "state_version", 3)
    path = save_changed(tmp_path, backbone={"value": 1.0})
    monkeypatch.setattr(Backbone, "state_version", 1)

    with pytest.raises(ValueError, match="'backbone'.*newer"):
        Checkpointer.load_checkpoint(path)


def test_every_component_that_cannot_take_its_state_is_reported(
        tmp_path, monkeypatch,
):
    path = save_changed(tmp_path, backbone={"value": 1.0}, head={})
    monkeypatch.setattr(Backbone, "state_version", 2)
    monkeypatch.setattr(Head, "state_version", 2)

    with pytest.raises(ValueError) as raised:
        Checkpointer.load_checkpoint(path)

    assert "'backbone'" in str(raised.value)
    assert "'head'" in str(raised.value)


def test_reinit_keeps_a_mismatched_component_as_freshly_built(
        tmp_path, monkeypatch,
):
    path = save_changed(tmp_path, backbone={"value": 1.0}, head={"value": 2.0})
    monkeypatch.setattr(Backbone, "state_version", 2)

    with pytest.warns(RuntimeWarning, match="backbone"):
        restored = Checkpointer.load_checkpoint(path, on_mismatch="reinit")

    assert value_of(resource_named(restored, "backbone")) == 1.0
    assert value_of(resource_named(restored, "head")) == 2.0


def test_reinit_by_name_covers_only_the_components_named(tmp_path, monkeypatch):
    path = save_changed(tmp_path, backbone={"value": 1.0}, head={"value": 2.0})
    monkeypatch.setattr(Backbone, "state_version", 2)
    monkeypatch.setattr(Head, "state_version", 2)

    with pytest.raises(ValueError, match="'head'"):
        Checkpointer.load_checkpoint(path, on_mismatch={"backbone"})

    with pytest.warns(RuntimeWarning):
        restored = Checkpointer.load_checkpoint(
            path, on_mismatch={"backbone", "head"},
        )
    assert value_of(resource_named(restored, "backbone")) == 1.0


def test_every_component_that_cannot_be_built_is_reported(tmp_path, monkeypatch):
    path = save(tmp_path, backbone={}, fragile={})
    # Both classes have since declared a prerequisite the checkpoint never
    # held, as `@requires_resource("tokenizer")` would.
    monkeypatch.setattr(
        Backbone, "required_resources", ("tokenizer",), raising=False,
    )
    monkeypatch.setattr(
        Fragile, "required_resources", ("tokenizer",), raising=False,
    )

    with pytest.raises(ValueError) as raised:
        Checkpointer.load_checkpoint(path)

    message = str(raised.value)
    assert "'backbone' requires 'tokenizer'" in message
    assert "'fragile' requires 'tokenizer'" in message


def test_a_failed_state_migration_keeps_the_migrated_constructor_arguments(
        tmp_path, monkeypatch,
):
    path = save(tmp_path, scaled={"value": 1.0})
    monkeypatch.setitem(CONSTRUCTOR_NEEDS_SCALE, "scaled", True)
    monkeypatch.setattr(Scaled, "state_version", 2)
    monkeypatch.setattr(
        Scaled,
        "migrate_init_args",
        classmethod(lambda cls, version, init_args: {
            "args": init_args["args"],
            "kwargs": {**init_args["kwargs"], "scale": 3.0},
        }),
    )

    with pytest.warns(RuntimeWarning, match="scaled"):
        restored = Checkpointer.load_checkpoint(path, on_mismatch="reinit")

    assert resource_named(restored, "scaled").scale == 3.0


def test_constructor_arguments_that_cannot_migrate_are_never_reinitialized(
        tmp_path, monkeypatch,
):
    path = save(tmp_path, scaled={"value": 1.0})
    monkeypatch.setattr(Scaled, "state_version", 2)

    def refuse(cls, version, init_args):
        raise ValueError("no way to rebuild scaled's arguments")

    monkeypatch.setattr(Scaled, "migrate_init_args", classmethod(refuse))

    with pytest.raises(ValueError, match="no way to rebuild"):
        Checkpointer.load_checkpoint(path, on_mismatch="reinit")


# -- a checkpoint says what it holds, not where to read it from ---------------------


def point_record_at(path, name, file, *, in_session_record=False):
    if in_session_record:
        record = torch.load(f"{path}/session.pt", weights_only=True)
        record["components"][name]["file"] = file
        torch.save(record, f"{path}/session.pt")
        return
    with open(f"{path}/manifest.json") as manifest_file:
        manifest = json.load(manifest_file)
    manifest["components"][name]["file"] = file
    with open(f"{path}/manifest.json", "w") as manifest_file:
        json.dump(manifest, manifest_file)


@pytest.mark.parametrize("outside", ["absolute", "relative"])
def test_a_recorded_file_outside_the_checkpoint_is_never_read(tmp_path, outside):
    path = save(tmp_path, backbone={"value": 1.0})
    torch.save({"value": torch.tensor([42.0])}, tmp_path / "elsewhere.pt")
    file = (
        str(tmp_path / "elsewhere.pt")
        if outside == "absolute"
        else "components/../../elsewhere.pt"
    )

    point_record_at(path, "backbone", file)
    with pytest.raises(ValueError, match="can only be at 'components/backbone.pt'"):
        Checkpointer.load_component_state(path, "backbone")

    point_record_at(path, "backbone", file, in_session_record=True)
    with pytest.raises(ValueError, match="can only be at 'components/backbone.pt'"):
        Checkpointer.load_checkpoint(path)


def test_a_symlinked_component_file_is_never_read(tmp_path):
    path = save(tmp_path, backbone={"value": 1.0})
    component = tmp_path / "checkpoint" / "components" / "backbone.pt"
    shutil.move(component, tmp_path / "moved.pt")
    os.symlink(tmp_path / "moved.pt", component)

    with pytest.raises(ValueError, match="symlink"):
        Checkpointer.load_checkpoint(path)
    with pytest.raises(ValueError, match="symlink"):
        Checkpointer.load_component_state(path, "backbone")


# -- a consumer keeps the instance it was given --------------------------------------


def test_a_later_sibling_does_not_make_a_given_instance_ambiguous(tmp_path):
    session = build(tmp_path, **{"backbone#b": {"value": 2.0}, "head": {}})
    # Activated after head was given backbone#b: `backbone` now has two
    # instances, but head holds, and uses, the one it was given.
    session.activate_component("backbone#a", {"value": 1.0})

    assert "Resource.backbone#b" in session.execution_graph()
    assert {"head", "backbone#b"} <= session.rank_parallel_names()

    restored = Checkpointer.load_checkpoint(save(tmp_path, session))

    head = resource_named(restored, "head")
    assert head.get_dependency("backbone").name == "backbone#b"
    assert "Resource.backbone#b" in restored.execution_graph()
    with restored:
        assert list(restored) == [1, 2]
