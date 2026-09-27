"""`import_components`: another run's components, taken in as this session's.

Each test writes a source run to disk, then builds a session that imports
from it. The fine-tuning built-ins are covered in
`components/builtin/test_fine_tuning.py`; these tests are about the import
itself: naming, wiring, restore, and what is refused.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.test_utils import (
    configurator_for,
    has_resource_named,
    make_config,
    resource_named,
)
from training_framework.components import (
    ComponentDependencyError,
    ModuleResource,
    Resource,
    Step,
    activates,
    requires_resource,
    resource,
    singleton,
    step,
)
from training_framework.components.builtin import Checkpointer
from training_framework.engine import TrainingEngine, load_session_for_worker
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


class KeywordModel(ModuleResource):
    """Built with a keyword argument, which no config key can express."""

    def __init__(self, config=None, *, width=5):
        super().__init__(config)
        self.linear = nn.Linear(4, width)


@requires_resource("net")
class Consumer(ModuleResource):
    """This session's own model, using whatever fills `net`."""

    def __init__(self, config=None):
        super().__init__(config)
        self.net = self.get_dependency("net")

    def forward(self, x):
        return self.net(x)


@requires_resource("imp_block")
class BlockUser(ModuleResource):
    """Asks for `imp_block` by its implementation name."""

    def __init__(self, config=None):
        super().__init__(config)
        self.block = self.get_dependency("imp_block")


@requires_resource("net")
class NeedyBlock(Block):
    """A block of this session's that itself needs `net`."""


class Tokenizer(Resource):
    """Not a module, and needs its setup: it holds nothing until then."""

    def __init__(self, config=None):
        self.vocabulary = None

    def setup(self, session):
        self.vocabulary = ["<pad>", "a", "b"]

    def teardown(self, session):
        self.vocabulary = None


@requires_resource("tokenizer")
class TokenizedBlock(Block):
    def __init__(self, config=None):
        super().__init__(config)
        self.tokenizer = self.get_dependency("tokenizer")


@singleton
class Only(Resource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("only")
class NeedsOnly(Block):
    pass


class CompanionStep(Step):
    def run(self, session):
        return None


@activates("imp_companion_step")
class WithCompanion(Block):
    pass


def _register():
    resource("imp_block")(Block)
    resource("imp_composite")(Composite)
    resource("imp_keyword_model")(KeywordModel)
    resource("imp_consumer")(Consumer)
    resource("imp_block_user")(BlockUser)
    resource("imp_needy_block")(NeedyBlock)
    resource("imp_tokenizer")(Tokenizer)
    resource("imp_tokenized_block")(TokenizedBlock)
    resource("only")(Only)
    resource("imp_needs_only")(NeedsOnly)
    step("imp_companion_step")(CompanionStep)
    resource("imp_with_companion")(WithCompanion)


@pytest.fixture(autouse=True)
def _components():
    _register()


# -- helpers -----------------------------------------------------------------------


def source_run(tmp_path, bindings, name="source", **components):
    """Write a run wired as `bindings` says; return its path and session."""
    config = make_config(tmp_path / name, seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = bindings
    config.update(components)
    session = TrainingSession(config)
    return Checkpointer.save_checkpoint(session, tmp_path / f"{name}-ckpt"), session


def composite_run(tmp_path, name="source"):
    return source_run(
        tmp_path, {"model": "imp_composite", "block": "imp_block"}, name,
        imp_composite={}, imp_block={},
    )


def importing_config(tmp_path, imports, bindings=None, **components):
    config = make_config(tmp_path / "run", seed=11)
    config["session_config"]["show_execution_graph"] = False
    config["import_components"] = imports
    config["component_bindings"] = bindings or {}
    config.update(components)
    return config


def importing(tmp_path, imports, bindings=None, **components):
    return TrainingSession(importing_config(tmp_path, imports, bindings, **components))


def imported_as_model(path, **options):
    return {"pretrained": {"checkpoint": str(path), "role": "model", **options}}


def weights(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def assert_same_weights(module, reference):
    expected = weights(reference)
    actual = weights(module)
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name])


# -- names -------------------------------------------------------------------------


def test_imported_instances_keep_their_source_names(tmp_path):
    path, source = composite_run(tmp_path)

    session = importing(tmp_path, imported_as_model(path))

    composite = resource_named(session, "imp_composite")
    assert composite.block is resource_named(session, "imp_block")
    assert_same_weights(composite, resource_named(source, "imp_composite"))


def test_a_suffix_renames_every_imported_instance(tmp_path):
    path, source = composite_run(tmp_path)

    session = importing(tmp_path, imported_as_model(path, suffix="pre"))

    composite = resource_named(session, "imp_composite#pre")
    assert composite.block is resource_named(session, "imp_block#pre")
    assert_same_weights(composite, resource_named(source, "imp_composite"))
    # Its wiring was renamed with it, so it saves and restores.
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")
    restored = Checkpointer.load_checkpoint(saved)
    assert_same_weights(resource_named(restored, "imp_composite#pre"), composite)


def test_two_imports_of_one_implementation_need_suffixes(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    imports = {
        "a": {"checkpoint": str(first), "role": "model", "suffix": "a"},
        "b": {"checkpoint": str(second), "role": "net", "suffix": "b"},
    }

    session = importing(tmp_path, imports)

    assert resource_named(session, "imp_block#a") is not resource_named(session, "imp_block#b")


def test_a_clash_with_another_import_suggests_a_suffix(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    imports = {
        "a": {"checkpoint": str(first), "role": "model"},
        "b": {"checkpoint": str(second), "role": "net"},
    }

    with pytest.raises(ValueError, match=r"import_components.b imports 'imp_block', but import_components.a imports one too.*suffix"):
        importing(tmp_path, imports)


def test_a_clash_with_a_configured_instance_suggests_a_suffix(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"imports 'imp_block', but this session configures one.*suffix.*bind"):
        importing(tmp_path, imported_as_model(path), imp_block={})


# -- restore -------------------------------------------------------------------------


def test_a_component_built_with_keyword_arguments_is_rebuilt_from_them(tmp_path):
    config = make_config(tmp_path / "source", seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"model": "imp_block"}
    config["imp_block"] = {}
    source = TrainingSession(config)
    source.register_resource(KeywordModel({}, width=3))
    path = Checkpointer.save_checkpoint(source, tmp_path / "source-ckpt")

    session = importing(tmp_path, {
        "kw": {"checkpoint": str(path), "resource": "imp_keyword_model", "role": "model"},
    })

    assert resource_named(session, "imp_keyword_model").linear.out_features == 3


def test_an_older_version_is_migrated_then_renamed(tmp_path):
    # The source's version keeps the wiring it recorded under an old key; the
    # current one migrates it into `linked`. Renaming before the migration
    # would find no `linked`, and restore's wiring check would then refuse
    # the state.
    @requires_resource("block")
    class CompositeV1(Composite):
        def get_state(self):
            state = dict(super().get_state())
            state["links"] = state.pop("linked")
            return state

    resource("imp_composite", overwrite=True)(CompositeV1)
    path, source = composite_run(tmp_path)

    @requires_resource("block")
    class CompositeV2(Composite):
        state_version = 2

        @classmethod
        def migrate_state(cls, from_version, state):
            state = dict(state)
            state["linked"] = state.pop("links")
            state["state_dict"] = {
                key: value * 2 if key == "scale" else value
                for key, value in state["state_dict"].items()
            }
            return state

    resource("imp_composite", overwrite=True)(CompositeV2)
    session = importing(tmp_path, imported_as_model(path, suffix="pre"))

    composite = resource_named(session, "imp_composite#pre")
    assert composite.get_state()["linked"] == {"block": "imp_block#pre"}
    torch.testing.assert_close(
        composite.scale, 2 * resource_named(source, "imp_composite").scale,
    )


def test_a_prerequisite_that_needs_setup_is_set_up(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "imp_tokenized_block", "tokenizer": "imp_tokenizer"},
        imp_tokenized_block={}, imp_tokenizer={},
    )
    session = importing(tmp_path, imported_as_model(path))
    tokenizer = resource_named(session, "imp_tokenizer")

    assert resource_named(session, "imp_tokenized_block").tokenizer is tokenizer
    with session:
        assert tokenizer.vocabulary == ["<pad>", "a", "b"]
    assert tokenizer.vocabulary is None


def test_a_state_that_cannot_be_migrated_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    @requires_resource("block")
    class CompositeV2(Composite):
        state_version = 2

    resource("imp_composite", overwrite=True)(CompositeV2)
    with pytest.raises(ValueError, match="no migrate_state"):
        importing(tmp_path, imported_as_model(path))


# -- wiring --------------------------------------------------------------------------


def test_bind_gives_the_import_this_sessions_component(tmp_path):
    path, _ = composite_run(tmp_path)

    session = importing(
        tmp_path, imported_as_model(path, bind={"block": "imp_needy_block#mine"}),
        {"net": "imp_block#other"},
        **{"imp_needy_block#mine": {}, "imp_block#other": {}},
    )

    composite = resource_named(session, "imp_composite")
    assert composite.block is resource_named(session, "imp_needy_block#mine")
    # The source's block was never imported.
    assert not has_resource_named(session, "imp_block")
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")
    Checkpointer.load_checkpoint(saved)


def test_a_bind_key_the_import_never_reaches_is_refused(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "imp_composite", "block": "imp_block"},
        imp_composite={}, imp_block={}, imp_tokenizer={},
    )

    with pytest.raises(ValueError, match=r"bind names \['imp_tokenizer'\], which 'imp_composite' is not wired to.*\['imp_block', 'imp_composite'\]"):
        importing(tmp_path, imported_as_model(path, bind={"imp_tokenizer": "imp_tokenizer"}))


def test_a_prerequisite_declared_since_the_source_run_must_be_wired(tmp_path):
    path, _ = source_run(tmp_path, {"model": "imp_block"}, imp_block={})

    @requires_resource("tokenizer")
    class GrownBlock(Block):
        pass

    resource("imp_block", overwrite=True)(GrownBlock)
    # Reported through restore, which gathers every problem into one error.
    with pytest.raises(ValueError, match=r"'imp_block' declares \['tokenizer'\], which the run it comes from never wired"):
        importing(tmp_path, imported_as_model(path), imp_tokenizer={})

    session = importing(
        tmp_path, imported_as_model(path),
        {"imp_block": {"tokenizer": "imp_tokenizer"}}, imp_tokenizer={},
    )
    assert resource_named(session, "imp_block").get_dependency("tokenizer") is (
        resource_named(session, "imp_tokenizer")
    )


def test_a_binding_for_wiring_the_source_already_made_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"wires \['block'\] of imported component 'imp_composite', which the run it comes from already wired.*`bind`"):
        importing(
            tmp_path, imported_as_model(path),
            {"imp_composite": {"block": "imp_block#mine"}}, **{"imp_block#mine": {}},
        )


def test_a_binding_for_a_name_the_import_does_not_declare_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"wires \['blok'\] of imported component 'imp_composite', which Composite does not declare"):
        importing(
            tmp_path, imported_as_model(path),
            {"imp_composite": {"blok": "imp_block#mine"}}, **{"imp_block#mine": {}},
        )


def test_what_the_import_is_wired_to_cannot_need_the_import(tmp_path):
    path, _ = composite_run(tmp_path)
    imports = {"p": {
        "checkpoint": str(path), "role": "net",
        "bind": {"block": "imp_needy_block#mine"},
    }}

    with pytest.raises(ComponentDependencyError, match=r"import_components.p: 'imp_needy_block#mine' is wired to the import.*imp_needy_block#mine -> imp_composite"):
        importing(
            tmp_path, imports, {"model": "imp_consumer"},
            imp_consumer={}, **{"imp_needy_block#mine": {}},
        )


def test_a_role_already_bound_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match="role binds 'model', which is already bound to 'imp_consumer'"):
        importing(tmp_path, imported_as_model(path), {"model": "imp_consumer"})


def test_an_import_fills_this_sessions_dependencies_only_through_a_binding(tmp_path):
    path, _ = composite_run(tmp_path)

    # By exact name.
    with pytest.raises(ComponentDependencyError, match=r"'imp_block_user' asks for 'imp_block' and would be given 'imp_block'.*no binding names it"):
        importing(
            tmp_path, imported_as_model(path), {"net": "imp_block_user"},
            imp_block_user={},
        )
    # As the sole instance of its implementation.
    with pytest.raises(ComponentDependencyError, match=r"would be given 'imp_block#pre'.*no binding names it"):
        importing(
            tmp_path, imported_as_model(path, suffix="pre"),
            {"net": "imp_block_user"}, imp_block_user={},
        )
    # Named by a binding: accepted.
    session = importing(
        tmp_path, imported_as_model(path, suffix="pre"),
        {"net": "imp_block_user", "imp_block_user": {"imp_block": "imp_block#pre"}},
        imp_block_user={},
    )
    assert resource_named(session, "imp_block_user").block is (
        resource_named(session, "imp_block#pre")
    )


# -- refused imports -----------------------------------------------------------------


def test_a_singleton_is_refused(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "imp_needs_only"}, imp_needs_only={}, only={},
    )

    with pytest.raises(ComponentDependencyError, match=r"'only' is @singleton.*`bind`"):
        importing(tmp_path, imported_as_model(path))


def test_a_component_with_companions_is_refused(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "imp_with_companion"}, imp_with_companion={},
    )

    with pytest.raises(ComponentDependencyError, match=r"'imp_with_companion' activates \['imp_companion_step'\]"):
        importing(tmp_path, imported_as_model(path))


def test_only_resources_can_be_imported(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ComponentDependencyError, match="'checkpointer' is a Hook; only resources can be imported"):
        importing(tmp_path, {"p": {"checkpoint": str(path), "resource": "checkpointer"}})


def test_a_legacy_single_file_source_is_refused(tmp_path):
    _, source = composite_run(tmp_path)
    legacy = tmp_path / "legacy.pt"
    torch.save(source, legacy)

    with pytest.raises(ValueError, match="single-file checkpoint, written before 0.5.0, and cannot be imported"):
        importing(tmp_path, imported_as_model(legacy))


@pytest.mark.parametrize("imports, message", [
    ({"p": {"resource": "model"}}, "import_components.p.checkpoint is required"),
    ({"p": {"checkpoint": "x", "suffix": "a-b"}}, "suffix must be one or more letters"),
    ({"p": {"checkpoint": "x", "role": "net#a"}}, "role must be a role name"),
    ({"p": {"checkpoint": "x", "rename": "a"}}, r"unknown keys \['rename'\]"),
    (["p"], "import_components must be a mapping"),
])
def test_a_malformed_import_is_refused(tmp_path, imports, message):
    with pytest.raises(ValueError, match=message):
        importing(tmp_path, imports)


# -- bindings for components the session does not hold ---------------------------------


def test_a_binding_for_a_misspelt_instance_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match=r"wires 'imp_composite#pree', which this session does not hold.*\['imp_composite#pre'\]"):
        importing(
            tmp_path, imported_as_model(path, suffix="pre"),
            {"imp_composite#pree": {"block": "imp_block"}},
        )


def test_a_binding_for_a_component_never_activated_is_refused(tmp_path):
    config = make_config(tmp_path, seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {
        "model": "imp_block",
        "imp_consumer": {"net": "imp_block"},
    }
    config["imp_block"] = {}
    session = TrainingSession(config)

    # Checked once nothing more can be activated by hand: on entering the
    # session (or earlier, when asked; the engine asks before the ranks start).
    with pytest.raises(ValueError, match="wires 'imp_consumer', which this session does not hold"):
        session.check_component_bindings()


def stale_binding_config(tmp_path):
    config = make_config(tmp_path, seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {
        "model": "imp_block",
        "imp_consumer": {"net": "imp_block"},
    }
    config["imp_block"] = {}
    return config


def test_entering_a_session_checks_its_bindings(tmp_path):
    session = TrainingSession(stale_binding_config(tmp_path))

    with pytest.raises(ValueError, match="wires 'imp_consumer', which this session does not hold"):
        with session:
            pass


def test_the_engine_checks_bindings_before_starting_ranks(tmp_path, monkeypatch):
    configurator = configurator_for(tmp_path, monkeypatch, stale_binding_config(tmp_path))

    with pytest.raises(ValueError, match="wires 'imp_consumer', which this session does not hold"):
        with TrainingEngine(configurator):
            pass


def test_a_binding_for_a_component_activated_by_hand_is_accepted(tmp_path):
    config = make_config(tmp_path, seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {
        "model": "imp_block",
        "imp_consumer": {"net": "imp_block"},
    }
    config["imp_block"] = {}
    session = TrainingSession(config)

    session.activate_component("imp_consumer", {})

    session.check_component_bindings()
    assert resource_named(session, "imp_consumer").net is resource_named(session, "imp_block")


# -- the run afterwards --------------------------------------------------------------


def test_an_extension_cannot_change_the_imports(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, imported_as_model(path))

    with pytest.raises(ValueError, match="does not allow changes to: import_components"):
        session.apply_extension_overrides(["import_components.pretrained.suffix=x"])


def test_a_fresh_run_from_the_stored_config_imports_again(tmp_path):
    path, source = composite_run(tmp_path)
    session = importing(tmp_path, imported_as_model(path))
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    stored = Checkpointer.read_manifest(saved)["config"]
    assert stored["import_components"] == imported_as_model(path)
    again = TrainingSession(stored)

    assert_same_weights(
        resource_named(again, "imp_composite"),
        resource_named(source, "imp_composite"),
    )


# -- after the session is built ------------------------------------------------------


def ranked_config(tmp_path, path, **ddp):
    config = importing_config(tmp_path, imported_as_model(path))
    config["ddp"] = {
        "world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1",
        "master_port": "12399", **ddp,
    }
    return config


def test_a_rank_worker_keeps_the_import_role(tmp_path):
    path, _ = composite_run(tmp_path)
    parent = TrainingSession(ranked_config(tmp_path, path))

    rank_one = load_session_for_worker(parent.get_state(), rank=1)

    assert rank_one.resolve_component_name("model") == "imp_composite"
    assert has_resource_named(rank_one, "imp_composite")
    assert has_resource_named(rank_one, "imp_block")


def test_rank_pruning_resolves_an_import_role(tmp_path):
    path, _ = source_run(
        tmp_path, {"model": "imp_tokenized_block", "tokenizer": "imp_tokenizer"},
        imp_tokenized_block={}, imp_tokenizer={},
    )
    config = importing_config(
        tmp_path,
        {"tok": {"checkpoint": str(path), "resource": "tokenizer", "role": "tok"}},
        {"model": "imp_block"},
        imp_block={},
    )
    config["ddp"] = {
        "world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1",
        "master_port": "12399", "rank_zero_components": ["tok"],
    }

    # The worker resolves `tok` as the parent did, and leaves it to rank 0.
    rank_one = load_session_for_worker(TrainingSession(config).get_state(), rank=1)

    assert not has_resource_named(rank_one, "imp_tokenizer")
    assert has_resource_named(rank_one, "imp_block")


def test_a_resumed_session_keeps_the_import_role(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, imported_as_model(path))
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    resumed = Checkpointer.load_checkpoint(saved)

    assert resumed.resolve_component_name("model") == "imp_composite"
    assert isinstance(Checkpointer.load_component(saved, "model"), Composite)


def test_the_import_role_is_in_the_manifest(tmp_path):
    path, source = composite_run(tmp_path)
    # Nothing in this session asks for `source`: only the import binds it.
    session = importing(tmp_path, {"p": {"checkpoint": str(path), "role": "source"}})
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    state = Checkpointer.load_component_state(saved, "source")
    # The composite's own weight; its block saves its own.
    assert set(state["state_dict"]) == {"scale"}

    # A run that imported can itself be imported from, by that role.
    again = importing(tmp_path / "again", {"again": {
        "checkpoint": str(saved), "resource": "source", "role": "model",
    }})
    assert_same_weights(
        resource_named(again, "imp_composite"),
        resource_named(source, "imp_composite"),
    )


def test_each_imported_component_records_its_import(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, imported_as_model(path))
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    components = Checkpointer.read_manifest(saved)["components"]
    assert components["imp_composite"]["imported_by"] == "import_components.pretrained"
    assert components["imp_block"]["imported_by"] == "import_components.pretrained"
    assert "imported_by" not in components["logger"]

    # Imported again, it is this import's: the source's record is replaced.
    again = importing(tmp_path / "again", {"again": {
        "checkpoint": str(saved), "role": "model",
    }})
    saved_again = Checkpointer.save_checkpoint(again, tmp_path / "saved-again")
    components = Checkpointer.read_manifest(saved_again)["components"]
    assert components["imp_block"]["imported_by"] == "import_components.again"


def test_a_component_activated_later_reaches_an_import_only_through_a_binding(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, imported_as_model(path, suffix="pre"))

    with pytest.raises(ComponentDependencyError, match=r"'imp_block_user' asks for 'imp_block' and would be given 'imp_block#pre'"):
        session.activate_component("imp_block_user", {})
    # Refused before anything was built.
    assert not has_resource_named(session, "imp_block_user")

    # Also once resumed: which instances are imported is saved.
    fresh = importing(tmp_path, imported_as_model(path, suffix="pre"))
    resumed = Checkpointer.load_checkpoint(
        Checkpointer.save_checkpoint(fresh, tmp_path / "saved"),
    )
    with pytest.raises(ComponentDependencyError, match="would be given 'imp_block#pre'"):
        resumed.activate_component("imp_block_user", {})


def test_a_binding_to_an_unsuffixed_imported_name_is_ambiguous(tmp_path):
    path, _ = composite_run(tmp_path)

    # Without the import, `net` would get a new `imp_block`.
    with pytest.raises(ComponentDependencyError, match=r"'imp_consumer' is bound to 'imp_block' for 'net'.*new one or the imported one.*`role`.*`suffix`"):
        importing(
            tmp_path, {"p": {"checkpoint": str(path)}},
            {"model": "imp_consumer", "net": "imp_block"}, imp_consumer={},
        )


def test_an_import_can_bind_to_what_an_earlier_import_brought(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    imports = {
        "a": {"checkpoint": str(first), "role": "model", "suffix": "a"},
        "b": {"checkpoint": str(second), "suffix": "b", "bind": {"block": "imp_block#a"}},
    }

    session = importing(tmp_path, imports)

    assert resource_named(session, "imp_composite#b").block is (
        resource_named(session, "imp_block#a")
    )
    assert not has_resource_named(session, "imp_block#b")


def test_binding_an_earlier_imports_unsuffixed_instance_is_ambiguous(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    imports = {
        "a": {"checkpoint": str(first), "role": "model"},
        "b": {"checkpoint": str(second), "suffix": "b", "bind": {"block": "imp_block"}},
    }

    with pytest.raises(ComponentDependencyError, match=r"import_components.b.bind.block: 'imp_block' is also an instance import_components.a imports under its source name.*`suffix`"):
        importing(tmp_path, imports)


def test_an_earlier_import_is_never_bound_as_the_sole_instance(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    imports = {
        "a": {"checkpoint": str(first), "role": "model", "suffix": "pre"},
        "b": {"checkpoint": str(second), "suffix": "b", "bind": {"block": "imp_block"}},
    }

    # `imp_block#pre` is the only instance of `imp_block` there is, but the
    # bind did not name it: a new `imp_block` is meant, which needs a config.
    with pytest.raises(RuntimeError, match="Component 'imp_block' is required but defines a custom constructor"):
        importing(tmp_path, imports)


def test_a_per_consumer_binding_never_takes_an_earlier_import_as_the_sole_instance(tmp_path):
    tokenizer_run, _ = source_run(
        tmp_path, {"model": "imp_tokenized_block", "tokenizer": "imp_tokenizer"},
        "tokenized", imp_tokenized_block={}, imp_tokenizer={},
    )
    block_run, _ = source_run(tmp_path, {"model": "imp_block"}, "block", imp_block={})

    @requires_resource("tokenizer")
    class GrownBlock(Block):
        pass

    resource("imp_block", overwrite=True)(GrownBlock)
    imports = {
        "tok": {"checkpoint": str(tokenizer_run), "resource": "imp_tokenizer", "suffix": "pre"},
        "blk": {"checkpoint": str(block_run), "role": "model"},
    }

    with pytest.raises(RuntimeError, match="Component 'imp_tokenizer' is required but defines a custom constructor"):
        importing(tmp_path, imports, {"imp_block": {"tokenizer": "imp_tokenizer"}})


def test_an_ambiguous_bind_target_names_the_bind_key(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ComponentDependencyError, match=r"import_components.pretrained.bind.block: 'imp_block' could be any of \['imp_block#x', 'imp_block#y'\]; name the instance"):
        importing(
            tmp_path, imported_as_model(path, bind={"block": "imp_block"}),
            **{"imp_block#x": {}, "imp_block#y": {}},
        )


def test_binding_to_what_a_later_import_brings_says_to_reorder(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")
    imports = {
        "b": {"checkpoint": str(second), "suffix": "b", "bind": {"block": "imp_block"}},
        "a": {"checkpoint": str(first), "role": "model"},
    }

    with pytest.raises(ComponentDependencyError, match=r"import_components.b: .*needs 'imp_block', which import_components.a imports.*List import_components.a before import_components.b"):
        importing(tmp_path, imports)


def test_a_bind_target_is_resolved_like_a_dependency(tmp_path):
    path, _ = composite_run(tmp_path)

    session = importing(
        tmp_path, imported_as_model(path, bind={"block": "imp_block"}),
        **{"imp_block#mine": {}},
    )

    # The only configured `imp_block` serves; no new one is built.
    assert resource_named(session, "imp_composite").block is (
        resource_named(session, "imp_block#mine")
    )
    assert not has_resource_named(session, "imp_block")


def test_a_bind_target_that_is_not_a_resource_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ValueError, match="import_components.pretrained.bind.block names 'logger', which cannot serve as a prerequisite"):
        importing(tmp_path, imported_as_model(path, bind={"block": "logger"}))


def test_an_imported_component_cannot_be_extended(tmp_path):
    path, _ = composite_run(tmp_path)
    session = importing(tmp_path, imported_as_model(path))

    with pytest.raises(ValueError, match=r"'imp_block' is imported \(import_components.pretrained\) and cannot be extended"):
        session.apply_extension_overrides(["imp_block.width=3"])


def test_a_per_consumer_target_of_an_import_is_resolved_once(tmp_path):
    # The source's composite reaches a tokenizer through its block, so the
    # import brings `imp_tokenizer#pre` along with it.
    path, _ = source_run(
        tmp_path,
        {"model": "imp_composite", "block": "imp_tokenized_block", "tokenizer": "imp_tokenizer"},
        imp_composite={}, imp_tokenized_block={}, imp_tokenizer={},
    )

    @requires_resource("block")
    @requires_resource("tokenizer")
    class GrownComposite(Composite):
        pass

    resource("imp_composite", overwrite=True)(GrownComposite)

    # `imp_tokenizer` means this session's own; the imported one is never a
    # candidate, also not when the restore wires the composite.
    session = importing(
        tmp_path, imported_as_model(path, suffix="pre"),
        {"imp_composite#pre": {"tokenizer": "imp_tokenizer"}},
        **{"imp_tokenizer#mine": {}},
    )

    assert resource_named(session, "imp_composite#pre").get_dependency("tokenizer") is (
        resource_named(session, "imp_tokenizer#mine")
    )


def test_a_bind_can_name_an_earlier_imports_role(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    session = importing(tmp_path, {
        "a": {"checkpoint": str(first), "role": "net", "suffix": "pre"},
        "b": {"checkpoint": str(second), "role": "model", "suffix": "b", "bind": {"block": "net"}},
    })
    assert resource_named(session, "imp_composite#b").block is (
        resource_named(session, "imp_composite#pre")
    )

    # Unsuffixed, the role's instance is ambiguous, as its name would be.
    with pytest.raises(ComponentDependencyError, match=r"import_components.b.bind.block: 'imp_composite' is also an instance import_components.a imports under its source name"):
        importing(tmp_path, {
            "a": {"checkpoint": str(first), "role": "net"},
            "b": {"checkpoint": str(second), "role": "model", "suffix": "b", "bind": {"block": "net"}},
        })


def test_a_bind_naming_a_later_imports_role_says_to_reorder(tmp_path):
    first, _ = composite_run(tmp_path, "first")
    second, _ = composite_run(tmp_path, "second")

    with pytest.raises(ComponentDependencyError, match=r"import_components.b.bind.block: 'net' is the role import_components.a binds.*List import_components.a before import_components.b"):
        importing(tmp_path, {
            "b": {"checkpoint": str(second), "role": "model", "suffix": "b", "bind": {"block": "net"}},
            "a": {"checkpoint": str(first), "role": "net", "suffix": "pre"},
        })


def test_a_bind_naming_the_imports_own_role_is_refused(tmp_path):
    path, _ = composite_run(tmp_path)

    with pytest.raises(ComponentDependencyError, match=r"import_components.pretrained.bind.block: 'model' is the role import_components.pretrained binds to its own resource"):
        importing(tmp_path, imported_as_model(path, bind={"block": "model"}))
