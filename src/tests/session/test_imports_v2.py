"""`import_components` keyed by the source's instance names.

Each key names an instance of the source run. What the import brings in is
renamed with its `instance_name`, or with the reserved `imported` when it has
none -- and an instance so named fills a dependency only when something
names it. The deprecated labelled form is covered by
`test_imports_legacy.py`.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from tests.test_utils import (
    has_resource_named,
    make_config,
    resource_named,
)
from training_framework.components import (
    ComponentDependencyError,
    ModuleResource,
    Resource,
    requires_resource,
    resource,
)
from training_framework.components.builtin import Checkpointer
from training_framework.engine import load_session_for_worker
from training_framework.session import TrainingSession


# -- components ------------------------------------------------------------------


class Block(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x)


@requires_resource("block")
class Composite(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.block = self.get_dependency("block")
        self.scale = nn.Parameter(torch.ones(5))

    def forward(self, x):
        return self.block(x) * self.scale


@requires_resource("net")
class Consumer(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.net = self.get_dependency("net")

    def forward(self, x):
        return self.net(x)


class Plain(Resource):
    """Built without configuration when nothing else fills its name."""

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("block")
class PlainComposite(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.block = self.get_dependency("block")


@requires_resource("kimp_plain")
class PlainUser(Resource):
    def __init__(self, config=None):
        self.plain = self.get_dependency("kimp_plain")

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("kimp_block")
class BlockUser(ModuleResource):
    """Asks for `kimp_block` by its implementation name."""

    def __init__(self, config=None):
        super().__init__(config)
        self.block = self.get_dependency("kimp_block")


@pytest.fixture(autouse=True)
def _components():
    resource("kimp_block")(Block)
    resource("kimp_composite")(Composite)
    resource("kimp_consumer")(Consumer)
    resource("kimp_block_user")(BlockUser)
    resource("kimp_plain")(Plain)
    resource("kimp_plain_composite")(PlainComposite)
    resource("kimp_plain_user")(PlainUser)


# -- helpers -----------------------------------------------------------------------


def source_run(tmp_path, bindings, name="source", **components):
    """Write a run wired as `bindings` says; return its path and session."""
    config = make_config(tmp_path / name, seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["role_bindings"] = bindings
    config.update(components)
    session = TrainingSession(config)
    return Checkpointer.save_checkpoint(session, tmp_path / f"{name}-ckpt"), session


def composite_run(tmp_path, name="source", block="kimp_block"):
    return source_run(
        tmp_path, {"model": "kimp_composite", "block": block}, name,
        kimp_composite={}, **{block: {}},
    )


def importing_config(tmp_path, imports, bindings=None, **components):
    config = make_config(tmp_path / "run", seed=11)
    config["session_config"]["show_execution_graph"] = False
    config["import_components"] = imports
    config["role_bindings"] = bindings or {}
    config.update(components)
    return config


def importing(tmp_path, imports, bindings=None, **components):
    with warnings.catch_warnings():
        # A keyed import never warns; a stray warning fails the test.
        warnings.simplefilter("error", DeprecationWarning)
        return TrainingSession(
            importing_config(tmp_path, imports, bindings, **components)
        )


NOT_NAMED = "Component 'kimp_block' is required but defines a custom constructor"
"""What a dependency on `kimp_block` gets when only `kimp_block#imported`
exists: it is not handed over, so a new `kimp_block` is needed."""


def composite_import(path, **options):
    return {"kimp_composite": {"checkpoint": str(path), **options}}


def assert_same_weights(module, reference):
    expected = {k: v.detach() for k, v in reference.state_dict().items()}
    actual = {k: v.detach() for k, v in module.state_dict().items()}
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name])


# -- names ------------------------------------------------------------------------


def test_an_import_without_instance_name_is_suffixed_imported(tmp_path):
    path, source = composite_run(tmp_path)

    session = importing(
        tmp_path, composite_import(path), {"model": "kimp_composite#imported"},
    )

    composite = resource_named(session, "kimp_composite#imported")
    assert composite.block is resource_named(session, "kimp_block#imported")
    assert_same_weights(composite, resource_named(source, "kimp_composite"))
    assert not has_resource_named(session, "kimp_composite")


def test_a_key_may_name_a_suffixed_instance(tmp_path):
    path, source = composite_run(tmp_path, block="kimp_block#x")

    session = importing(
        tmp_path, {"kimp_block#x": {"checkpoint": str(path)}},
        {"model": "kimp_block#imported_x"},
    )

    # The source's own suffix is kept after the import's.
    assert_same_weights(
        resource_named(session, "kimp_block#imported_x"),
        resource_named(source, "kimp_block#x"),
    )
    assert not has_resource_named(session, "kimp_composite#imported")


def test_instance_name_renames_every_imported_instance(tmp_path):
    path, source = composite_run(tmp_path, block="kimp_block#x")

    session = importing(
        tmp_path, composite_import(path, instance_name="pre"),
        {"model": "kimp_composite#pre"},
    )

    composite = resource_named(session, "kimp_composite#pre")
    assert composite.block is resource_named(session, "kimp_block#pre_x")
    assert_same_weights(composite, resource_named(source, "kimp_composite"))
    assert not has_resource_named(session, "kimp_composite#imported")


def test_one_source_name_imports_from_several_checkpoints_as_a_list(tmp_path):
    first, first_source = composite_run(tmp_path, "first")
    second, second_source = composite_run(tmp_path, "second")

    session = importing(tmp_path, {"kimp_composite": [
        {"checkpoint": str(first), "instance_name": "teacher"},
        {"checkpoint": str(second), "instance_name": "student"},
    ]}, {"model": "kimp_composite#student"})

    teacher = resource_named(session, "kimp_composite#teacher")
    student = resource_named(session, "kimp_composite#student")
    assert teacher.block is resource_named(session, "kimp_block#teacher")
    assert student.block is resource_named(session, "kimp_block#student")
    assert_same_weights(teacher, resource_named(first_source, "kimp_composite"))
    assert_same_weights(student, resource_named(second_source, "kimp_composite"))

    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")
    components = Checkpointer.read_manifest(saved)["components"]
    assert components["kimp_block#teacher"]["imported_by"] == "import_components.kimp_composite[0]"
    assert components["kimp_block#student"]["imported_by"] == "import_components.kimp_composite[1]"


def test_a_list_held_by_omegaconf_is_read_as_a_list(tmp_path):
    from omegaconf import OmegaConf

    first, _ = composite_run(tmp_path, "first")
    imports = OmegaConf.create({"kimp_composite": [
        {"checkpoint": str(first), "instance_name": "teacher"},
    ]})

    session = importing(tmp_path, imports, {"model": "kimp_composite#teacher"})

    assert has_resource_named(session, "kimp_block#teacher")


def test_list_entries_without_instance_name_clash(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    with pytest.raises(ValueError, match=r"import_components.kimp_composite\[1\] imports 'kimp_(block|composite)#imported', but import_components.kimp_composite\[0\] imports one too. Give the import an `instance_name:`"):
        importing(tmp_path, {"kimp_composite": [
            {"checkpoint": str(first)}, {"checkpoint": str(second)},
        ]})


def test_an_import_and_a_configured_instance_of_one_component_coexist(tmp_path):
    path, _ = composite_run(tmp_path)

    session = importing(
        tmp_path, composite_import(path), {"model": "kimp_composite#imported"},
        kimp_block={},
    )

    assert resource_named(session, "kimp_composite#imported").block is (
        resource_named(session, "kimp_block#imported")
    )
    assert resource_named(session, "kimp_block") is not (
        resource_named(session, "kimp_block#imported")
    )


def test_a_clash_with_a_configured_instance_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match="imports 'kimp_block#pre', but this session configures one"):
        importing(
            tmp_path, composite_import(path, instance_name="pre"),
            **{"kimp_block#pre": {}},
        )


# -- reaching the imported instances -------------------------------------------------


def test_an_unnamed_import_never_fills_a_dependency_by_itself(tmp_path):
    path, _ = composite_run(tmp_path)

    # `kimp_block#imported` is the only `kimp_block`, but nothing named it.
    with pytest.raises(RuntimeError, match=NOT_NAMED):
        importing(
            tmp_path, composite_import(path), {"model": "kimp_block_user"},
            kimp_block_user={},
        )


def test_an_unnamed_import_is_not_taken_instead_of_building_one(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "kimp_plain_composite", "block": "kimp_plain"},
        kimp_plain_composite={}, kimp_plain={},
    )

    session = importing(
        tmp_path, {"kimp_plain_composite": {"checkpoint": str(path)}},
        {"model": "kimp_plain_composite#imported"}, kimp_plain_user={},
    )

    # A new `kimp_plain` was built for the user; the import kept its own.
    assert resource_named(session, "kimp_plain_user").plain is (
        resource_named(session, "kimp_plain")
    )
    assert resource_named(session, "kimp_plain_composite#imported").block is (
        resource_named(session, "kimp_plain#imported")
    )


def test_a_renamed_instance_fills_a_dependency_as_the_sole_instance(tmp_path):
    path, _ = composite_run(tmp_path)

    session = importing(
        tmp_path, composite_import(path, instance_name="pre"),
        {"model": "kimp_block_user"}, kimp_block_user={},
    )

    assert resource_named(session, "kimp_block_user").block is (
        resource_named(session, "kimp_block#pre")
    )


def test_role_bindings_reach_an_imported_instance(tmp_path):
    path, _ = composite_run(tmp_path)

    session = importing(
        tmp_path, composite_import(path),
        {"model": "kimp_consumer", "net": "kimp_composite#imported"},
        kimp_consumer={},
    )

    assert resource_named(session, "kimp_consumer").net is (
        resource_named(session, "kimp_composite#imported")
    )
    assert session.resolve_component_name("net") == "kimp_composite#imported"


def test_a_consumers_own_binding_reaches_an_imported_instance(tmp_path):
    path, _ = composite_run(tmp_path)

    session = importing(
        tmp_path, composite_import(path), {"model": "kimp_block_user"},
        kimp_block_user={"dependencies_role_bindings": {"kimp_block": "kimp_block#imported"}},
    )

    assert resource_named(session, "kimp_block_user").block is (
        resource_named(session, "kimp_block#imported")
    )


def test_two_imported_instances_are_ambiguous_without_a_binding(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    with pytest.raises(ComponentDependencyError, match=r"kimp_block#a.*kimp_block#b"):
        importing(
            tmp_path,
            {"kimp_composite": [
                {"checkpoint": str(first), "instance_name": "a"},
                {"checkpoint": str(second), "instance_name": "b"},
            ]},
            {"model": "kimp_block_user"}, kimp_block_user={},
        )


def test_module_part_takes_its_source_from_an_import(tmp_path):
    path, source = composite_run(tmp_path)

    session = importing(
        tmp_path, composite_import(path),
        {
            "model": "kimp_consumer", "net": "module_part",
            "source": "kimp_composite#imported",
        },
        kimp_consumer={}, module_part={"submodule": "block"},
    )

    part = resource_named(session, "module_part")
    assert part.module is resource_named(session, "kimp_block#imported")
    x = torch.ones(2, 4)
    torch.testing.assert_close(
        resource_named(session, "kimp_consumer")(x),
        resource_named(source, "kimp_block")(x),
    )


# -- overwritten_dependencies ------------------------------------------------------


def test_overwritten_dependencies_can_name_an_earlier_imports_instance(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    session = importing(tmp_path, {"kimp_composite": [
        {"checkpoint": str(first)},
        {"checkpoint": str(second), "instance_name": "b",
         "overwritten_dependencies": {"block": "kimp_block#imported"}},
    ]}, {"model": "kimp_composite#b"})

    assert resource_named(session, "kimp_composite#b").block is (
        resource_named(session, "kimp_block#imported")
    )
    assert not has_resource_named(session, "kimp_block#b")


def test_overwritten_dependencies_do_not_take_an_unnamed_import_by_itself(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    with pytest.raises(RuntimeError, match=NOT_NAMED):
        importing(tmp_path, {"kimp_composite": [
            {"checkpoint": str(first)},
            {"checkpoint": str(second), "instance_name": "b",
             "overwritten_dependencies": {"block": "kimp_block"}},
        ]})


def test_overwritten_dependencies_take_an_earlier_import_as_the_sole_instance(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    session = importing(tmp_path, {"kimp_composite": [
        {"checkpoint": str(first), "instance_name": "a"},
        {"checkpoint": str(second), "instance_name": "b",
         "overwritten_dependencies": {"block": "kimp_block"}},
    ]}, {"model": "kimp_composite#b"})

    assert resource_named(session, "kimp_composite#b").block is (
        resource_named(session, "kimp_block#a")
    )


def test_overwritten_dependencies_on_a_later_import_say_to_reorder(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    with pytest.raises(ComponentDependencyError, match=r"import_components.kimp_composite\[0\].*needs 'kimp_block#b', which import_components.kimp_composite\[1\] imports.*List import_components.kimp_composite\[1\] before"):
        importing(tmp_path, {"kimp_composite": [
            {"checkpoint": str(first), "instance_name": "a",
             "overwritten_dependencies": {"block": "kimp_block#b"}},
            {"checkpoint": str(second), "instance_name": "b"},
        ]})


# -- the reserved suffix -----------------------------------------------------------


@pytest.mark.parametrize("name", ["kimp_block#imported", "kimp_block#imported_a"])
def test_a_configured_name_may_not_use_the_import_suffix(tmp_path, name):
    with pytest.raises(ValueError, match=f"Configured component '{name}' uses the instance suffix 'imported', which is reserved"):
        importing(tmp_path, {}, **{name: {}})


def test_a_component_activated_by_hand_may_not_use_the_import_suffix(tmp_path):
    session = importing(tmp_path, {})

    with pytest.raises(ValueError, match="Activated component 'kimp_block#imported' uses the instance suffix 'imported'"):
        session.activate_component("kimp_block#imported", {})
    assert not has_resource_named(session, "kimp_block#imported")


def test_a_component_registered_by_hand_may_not_use_the_import_suffix(tmp_path):
    session = importing(tmp_path, {})
    block = Block()
    block.name = "kimp_block#imported"

    with pytest.raises(ValueError, match="Registered component 'kimp_block#imported' uses the instance suffix 'imported'"):
        session.register_resource(block)


def test_names_restored_from_a_checkpoint_may_use_the_import_suffix(tmp_path):
    path, source = composite_run(tmp_path)
    session = importing(tmp_path, composite_import(path), {"model": "kimp_composite#imported"})
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    resumed = Checkpointer.load_checkpoint(saved)

    assert_same_weights(
        resource_named(resumed, "kimp_composite#imported"),
        resource_named(source, "kimp_composite"),
    )


def test_a_default_component_may_not_use_the_import_suffix(tmp_path):
    with pytest.raises(ValueError, match="Default component 'logger#imported' uses the instance suffix 'imported'"):
        importing(tmp_path, {}, {"logger": "logger#imported"})


def test_a_skipped_unnamed_import_is_named_in_the_error(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(RuntimeError, match=r"\['kimp_block#imported'\] fill a dependency only when named: bind one to use it"):
        importing(
            tmp_path, composite_import(path), {"model": "kimp_block_user"},
            kimp_block_user={},
        )


# -- after the session is built ------------------------------------------------------


def test_registering_by_hand_does_not_rewire_an_imported_component(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(
        tmp_path, composite_import(path, instance_name="pre"),
        {"model": "kimp_composite#pre", "block": "kimp_block"},
    )
    block = Block()
    block.name = "kimp_block"

    session.register_resource(block)

    composite = resource_named(session, "kimp_composite#pre")
    assert composite.get_dependency("block") is resource_named(session, "kimp_block#pre")
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")
    recorded = Checkpointer.read_manifest(saved)["components"]["kimp_composite#pre"]
    assert recorded["dependencies"] == {"block": "kimp_block#pre"}


def test_a_resumed_renamed_import_is_an_ordinary_instance(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(
        tmp_path, composite_import(path, instance_name="pre"),
        {"model": "kimp_composite#pre"},
    )
    resumed = Checkpointer.load_checkpoint(
        Checkpointer.save_checkpoint(session, tmp_path / "saved"),
    )

    resumed.activate_component("kimp_block_user", {})

    assert resource_named(resumed, "kimp_block_user").block is (
        resource_named(resumed, "kimp_block#pre")
    )


def test_a_rank_worker_resolves_a_sole_renamed_import_as_the_parent_did(tmp_path):
    path, _ = composite_run(tmp_path)
    config = importing_config(
        tmp_path, composite_import(path, instance_name="pre"),
        {"model": "kimp_block_user"}, kimp_block_user={},
    )
    config["ddp"] = {
        "world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1",
        "master_port": "12399",
    }
    parent = TrainingSession(config)

    rank_one = load_session_for_worker(parent.get_state(), rank=1)

    assert resource_named(rank_one, "kimp_block_user").block is (
        resource_named(rank_one, "kimp_block#pre")
    )


def test_a_resumed_session_resolves_imports_as_before(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, composite_import(path), {"model": "kimp_composite#imported"})
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    components = Checkpointer.read_manifest(saved)["components"]
    assert components["kimp_block#imported"]["imported_by"] == "import_components.kimp_composite"
    assert components["kimp_block#imported"]["imported_into_namespace"] is True
    assert "imported_into_namespace" not in components["logger"]

    resumed = Checkpointer.load_checkpoint(saved)
    # Still not handed over by itself...
    with pytest.raises(RuntimeError, match=NOT_NAMED):
        resumed.activate_component("kimp_block_user", {})
    assert not has_resource_named(resumed, "kimp_block_user")


def test_a_component_activated_later_reaches_a_renamed_import(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(
        tmp_path, composite_import(path, instance_name="pre"),
        {"model": "kimp_composite#pre"},
    )

    session.activate_component("kimp_block_user", {})

    assert resource_named(session, "kimp_block_user").block is (
        resource_named(session, "kimp_block#pre")
    )


def test_a_rank_worker_resolves_imports_as_the_parent_did(tmp_path):
    path, _ = composite_run(tmp_path)
    config = importing_config(
        tmp_path, composite_import(path),
        {"model": "kimp_consumer", "net": "kimp_composite#imported"},
        kimp_consumer={},
    )
    config["ddp"] = {
        "world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1",
        "master_port": "12399",
    }
    parent = TrainingSession(config)

    rank_one = load_session_for_worker(parent.get_state(), rank=1)

    assert resource_named(rank_one, "kimp_consumer").net is (
        resource_named(rank_one, "kimp_composite#imported")
    )
    assert has_resource_named(rank_one, "kimp_block#imported")


def test_rank_zero_components_may_name_an_import_by_a_binding(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "kimp_plain_composite", "block": "kimp_plain"},
        kimp_plain_composite={}, kimp_plain={},
    )
    config = importing_config(
        tmp_path, {"kimp_plain": {"checkpoint": str(path)}},
        {"model": "kimp_block", "tok": "kimp_plain#imported"}, kimp_block={},
    )
    config["ddp"] = {
        "world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1",
        "master_port": "12399", "rank_zero_components": ["tok"],
    }

    rank_one = load_session_for_worker(TrainingSession(config).get_state(), rank=1)

    assert not has_resource_named(rank_one, "kimp_plain#imported")
    assert has_resource_named(rank_one, "kimp_block")


def test_the_execution_graph_lists_what_was_imported(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, composite_import(path), {"model": "kimp_composite#imported"})

    graph = session.execution_graph()

    assert "IMPORTED" in graph
    assert "kimp_block#imported <- import_components.kimp_composite" in graph
    assert "kimp_composite#imported <- import_components.kimp_composite" in graph


def test_a_keyed_and_a_labelled_import_keep_their_own_rules(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    config = importing_config(tmp_path, {
        "old": {"checkpoint": str(first), "resource": "kimp_block", "suffix": "old"},
        "kimp_composite": {
            "checkpoint": str(second),
            "overwritten_dependencies": {"block": "kimp_block#old"},
        },
    }, {"model": "kimp_composite#imported"})
    with pytest.warns(DeprecationWarning):
        session = TrainingSession(config)
    assert resource_named(session, "kimp_composite#imported").block is (
        resource_named(session, "kimp_block#old")
    )

    # The labelled import's instance is the sole `kimp_block`, and is still
    # reached only by a binding.
    with pytest.raises(ComponentDependencyError, match="would be given 'kimp_block#old'.*no binding names it"):
        session.activate_component("kimp_block_user", {})


# -- the deprecated labelled form -------------------------------------------------


def test_a_labelled_import_of_a_keyed_import_is_held_apart(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, composite_import(path), {"model": "kimp_composite#imported"})
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    # The source's record says its instances were keyed; the labelled import
    # that brings them here decides for this session.
    config = importing_config(tmp_path / "again", {"old": {
        "checkpoint": str(saved), "resource": "model", "suffix": "s",
    }})
    with pytest.warns(DeprecationWarning):
        again = TrainingSession(config)
    assert has_resource_named(again, "kimp_block#s_imported")
    with pytest.raises(ComponentDependencyError, match="would be given 'kimp_block#s_imported'.*no binding names it"):
        again.activate_component("kimp_block_user", {})


def test_a_labelled_entry_warns(tmp_path):
    path, _ = composite_run(tmp_path)
    config = importing_config(
        tmp_path, {"pretrained": {"checkpoint": str(path), "role": "model"}},
    )

    with pytest.warns(DeprecationWarning, match=r"entries \['pretrained'\] use the deprecated form"):
        session = TrainingSession(config)
    assert session.resolve_component_name("model") == "kimp_composite"


def test_a_key_the_source_does_not_hold_is_read_as_a_label_with_a_warning(tmp_path):
    path, _ = composite_run(tmp_path)
    with pytest.warns(FutureWarning, match="'pretrained' is not an instance of the source run, so it is read as an import label"):
        session = TrainingSession(importing_config(
            tmp_path, {"pretrained": {"checkpoint": str(path)}},
        ))
    # The labelled form imports the source's `model`, under its source name.
    assert has_resource_named(session, "kimp_composite")


def test_the_label_fallback_warns_visibly_by_default(tmp_path):
    path, _ = composite_run(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        # Python's own default filters: a DeprecationWarning raised in library
        # code would be hidden by them; this one must not be.
        warnings.resetwarnings()
        warnings.simplefilter("default")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        TrainingSession(importing_config(
            tmp_path, {"kimp_compsite": {"checkpoint": str(path)}},
        ))

    messages = [str(w.message) for w in caught if w.category is FutureWarning]
    assert any("If the key is mistyped, fix it" in m for m in messages), messages


def test_a_role_key_is_refused_outside_a_list_too(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"'model' is a role of the source run, bound to \['kimp_composite'\]"):
        importing(tmp_path, {"model": {"checkpoint": str(path)}})


def test_a_missing_key_with_no_model_to_fall_back_on_is_refused(tmp_path):
    path, _ = source_run(tmp_path, {}, kimp_block={})

    with pytest.raises(ValueError, match=r"import_components.kimp_blok: the checkpoint has no component 'kimp_blok'; it holds .*'kimp_block'.*keyed by the source's instance name") as error:
        importing(tmp_path, {"kimp_blok": {"checkpoint": str(path)}})
    assert ".resource" not in str(error.value)


# -- what is refused ----------------------------------------------------------------


def test_a_key_the_source_does_not_hold_is_refused_when_keyed(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"import_components.kimp_missing: the checkpoint has no component 'kimp_missing'; it holds .*'kimp_composite'"):
        importing(tmp_path, {"kimp_missing": {"checkpoint": str(path), "instance_name": "a"}})


def test_a_key_that_is_a_role_of_the_source_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"'model' is a role of the source run, bound to \['kimp_composite'\]; key the import by the instance name"):
        importing(tmp_path, {"model": [{"checkpoint": str(path)}]})


@pytest.mark.parametrize("imports, message", [
    ({"kimp_composite": [{"checkpoint": "x", "suffix": "a"}]}, r"uses \['suffix'\], which only the deprecated labelled form accepts"),
    ({"kimp_composite": {"checkpoint": "x", "role": "model", "instance_name": "a"}}, r"unknown keys \['instance_name'\].*belongs to the keyed form"),
    ({"kimp_composite": {"checkpoint": "x", "instance_name": "a-b"}}, r"instance_name must be one or more letters, digits or underscores"),
    ({"kimp_composite": {"checkpoint": "x", "instance_name": "imported"}}, r"instance_name 'imported' is reserved"),
    ({"kimp_composite": {"checkpoint": "x", "instance_name": "imported_a"}}, r"instance_name 'imported_a' is reserved"),
    ({"kimp_composite": {"checkpoint": "x", "resourse": "y"}}, r"unknown keys \['resourse'\]"),
    ({"kimp_composite": []}, r"kimp_composite is an empty list"),
    ({"kimp_composite": ["x"]}, r"kimp_composite\[0\] must be a mapping"),
    ({"kimp_composite": {"instance_name": "a"}}, r"kimp_composite.checkpoint is required"),
])
def test_a_malformed_import_is_refused(tmp_path, imports, message):
    with pytest.raises(ValueError, match=message):
        importing(tmp_path, imports)
