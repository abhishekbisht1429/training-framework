"""Tests for listing a session's components under `resources`, `hooks` and
`steps`.

A group is only a layout: it activates what the flat top level would, and is
checked against the kind of each component it lists.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import yaml

from training_framework.components import (
    Hook,
    Resource,
    Step,
    requires_resource,
    resource,
    step,
)
from training_framework.engine import (
    Configurator,
    LaunchTopology,
    TrainingEngine,
    load_session_for_worker,
)
from training_framework.session import AnalysisSession, TrainingSession
from tests.test_utils import (
    COMPONENTS_PACKAGE,
    configurator_for,
    register_test_components,
)


def _config(tmp_path, entries):
    return {
        "session_config": {
            "rng_seed": 1,
            "sessions_dir": str(tmp_path),
            "max_iterations": 1,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
            "show_execution_graph": False,
        },
        **entries,
    }


def declare_components():
    """Register a resource `gc_dep` recording its configuration, and a step
    `gc_step` requiring it."""

    @resource("gc_dep")
    class Dependency(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.built_with = dict(config) if config is not None else None

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("gc_dep")
    @step("gc_step")
    class Consumer(Step):
        def run(self, session) -> None:
            pass


def _names(session):
    return {
        component.name
        for component in (
            session.get_all_resources()
            + session.get_all_hooks()
            + session.get_all_steps()
        )
    }


def _resource(session, name):
    (found,) = [
        component for component in session.get_all_resources()
        if component.name == name
    ]
    return found


def test_a_grouped_configuration_builds_what_its_flat_layout_builds(tmp_path):
    declare_components()
    flat = TrainingSession(_config(tmp_path / "flat", {
        "gc_dep#a": {"size": 1},
        "gc_dep#b": {"size": 2},
        "gc_step": {"dependencies_role_bindings": {"gc_dep": "gc_dep#b"}},
        "logger": {"log_every": 3},
    }))
    grouped = TrainingSession(_config(tmp_path / "grouped", {
        "resources": {"gc_dep#a": {"size": 1}, "gc_dep#b": {"size": 2}},
        "steps": {"gc_step": {"dependencies_role_bindings": {"gc_dep": "gc_dep#b"}}},
        "hooks": {"logger": {"log_every": 3}},
    }))

    assert _names(grouped) == _names(flat)
    assert grouped.execution_graph() == flat.execution_graph()
    assert _resource(grouped, "gc_dep#b").built_with == {"size": 2}


def test_grouped_and_flat_entries_may_be_mixed(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        "resources": {"gc_dep": {"size": 1}},
        "gc_step": {},
    }))

    assert {"gc_dep", "gc_step"} <= _names(session)


@pytest.mark.parametrize(
    ("entries", "match"),
    (
        pytest.param(
            {"resources": {"gc_step": {}}},
            r"'gc_step' is listed under 'resources', but it is a Step, not a "
            r"Resource\. List it under 'steps'",
            id="step-under-resources",
        ),
        pytest.param(
            {"steps": {"gc_dep#a": {}}},
            r"'gc_dep#a' is listed under 'steps', but it is a Resource, not a "
            r"Step\. List it under 'resources'",
            id="instance-under-steps",
        ),
        pytest.param(
            {"steps": {"logger": {}}},
            "it is a Hook, not a Step. List it under 'hooks'",
            id="hook-under-steps",
        ),
        pytest.param(
            {"role_bindings": {"gc_role": "gc_dep"}, "hooks": {"gc_dep": {}}},
            "it is a Resource, not a Hook",
            id="bound-implementation",
        ),
    ),
)
def test_a_component_under_the_wrong_group_is_refused(tmp_path, entries, match):
    declare_components()

    with pytest.raises(ValueError, match=match):
        TrainingSession(_config(tmp_path, entries))


@pytest.mark.parametrize(
    "entries",
    (
        pytest.param(
            {"resources": {"gc_dep": {}}, "steps": {"gc_dep": {}}},
            id="two-groups",
        ),
        pytest.param(
            {"resources": {"gc_dep": {}}, "gc_dep": {}},
            id="group-and-top-level",
        ),
    ),
)
def test_a_component_may_be_listed_once(tmp_path, entries):
    declare_components()

    with pytest.raises(ValueError, match="'gc_dep' is listed twice"):
        TrainingSession(_config(tmp_path, entries))


@pytest.mark.parametrize("grouped", (False, True), ids=("flat", "grouped"))
def test_an_empty_value_lists_a_component_with_no_settings(tmp_path, grouped):
    declare_components()
    entries = (
        {"resources": {"gc_dep": None}} if grouped else {"gc_dep": None}
    )

    session = TrainingSession(_config(tmp_path, entries))

    assert _resource(session, "gc_dep").built_with == {}


def test_an_empty_group_lists_nothing(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        "steps": None,
        "gc_dep": {},
        "gc_step": {},
    }))

    assert "gc_step" in _names(session)


@pytest.mark.parametrize(
    ("entries", "match"),
    (
        pytest.param({"steps": ["gc_step"]}, "'steps' must be a mapping", id="group"),
        pytest.param(
            {"steps": {"gc_step": 3}},
            "'steps.gc_step' is not a mapping",
            id="entry",
        ),
    ),
)
def test_a_group_and_its_entries_must_be_mappings(tmp_path, entries, match):
    declare_components()

    with pytest.raises(ValueError, match=match):
        TrainingSession(_config(tmp_path, entries))


def test_an_unregistered_name_in_a_group_gets_the_usual_diagnostics(tmp_path):
    declare_components()

    with pytest.raises(ValueError, match="No step, hook or resource registered"):
        TrainingSession(_config(tmp_path, {"steps": {"gc_nothing": {}}}))


def test_the_configuration_keeps_its_grouped_layout(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        "resources": {"gc_dep": {"size": 1}},
        "steps": {"gc_step": None},
    }))

    restored = TrainingSession.from_state(session.get_state())
    with restored:
        pass

    assert restored.full_config["resources"] == {"gc_dep": {"size": 1}}
    assert "gc_dep" not in restored.full_config
    assert _resource(restored, "gc_dep").built_with == {"size": 1}
    config_path = os.path.join(restored.session_config.session_dir, "config.yaml")
    with open(config_path) as config_file:
        assert yaml.safe_load(config_file)["resources"] == {"gc_dep": {"size": 1}}


def test_an_analysis_session_takes_groups(tmp_path):
    placeholder = tmp_path / "placeholder-checkpoint.pt"
    placeholder.touch()

    session = AnalysisSession(_config(tmp_path, {
        "resources": {
            "trained_model": {"model_checkpoint_path": str(placeholder)},
        },
    }))

    assert "trained_model" in _names(session)


# -- extension ------------------------------------------------------------------


def test_an_override_inside_a_group_extends_the_component(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        "hooks": {"logger": {"log_every": 1}},
        "gc_dep": {},
    }))

    session.apply_extension_overrides(("hooks.logger.log_every=4",))

    assert session.full_config["hooks"]["logger"]["log_every"] == 4
    assert "logger" not in session.full_config


def test_an_override_inside_a_group_reaches_an_unlisted_component(tmp_path):
    """`logger` is active by default; the override lists it where it says."""
    declare_components()
    session = TrainingSession(_config(tmp_path, {"gc_dep": {}}))

    session.apply_extension_overrides(("hooks.logger.log_every=4",))

    assert session.full_config["hooks"]["logger"]["log_every"] == 4
    assert "logger" not in session.full_config


@pytest.mark.parametrize(
    ("entries", "override", "match"),
    (
        pytest.param(
            {"hooks": {"logger": {"log_every": 1}}},
            "logger.log_every=4",
            r"'logger' is listed under 'hooks'; override 'hooks\.logger\.<key>'",
            id="flat-override-of-grouped",
        ),
        pytest.param(
            {"logger": {"log_every": 1}},
            "hooks.logger.log_every=4",
            r"'logger' is listed at the top level; override 'logger\.<key>'",
            id="grouped-override-of-flat",
        ),
    ),
)
def test_an_override_must_address_a_component_where_it_is_listed(
        tmp_path,
        entries,
        override,
        match,
):
    declare_components()
    session = TrainingSession(_config(tmp_path, entries))

    with pytest.raises(ValueError, match=match):
        session.apply_extension_overrides((override,))


@pytest.mark.parametrize("group", ["steps", "resources"])
def test_an_override_that_lists_a_component_under_the_wrong_group_is_refused(tmp_path, group):
    """An extension reads its configuration as a build does, so a hook listed
    under another kind's group is refused here too."""
    declare_components()
    session = TrainingSession(_config(tmp_path, {"gc_dep": {}}))

    with pytest.raises(ValueError, match=rf"'logger' is listed under '{group}', but it is a Hook, not a .*List it under 'hooks'"):
        session.apply_extension_overrides((f"{group}.logger.log_every=4",))
    # Nothing was stored: the right group still works.
    session.apply_extension_overrides(("hooks.logger.log_every=4",))
    assert session.full_config["hooks"]["logger"]["log_every"] == 4


@pytest.mark.parametrize(("group", "name"), [
    ("resources", "session_config"),
    ("steps", "role_bindings"),
    ("hooks", "import_components"),
])
def test_a_reserved_name_inside_a_group_is_refused_as_reserved(tmp_path, group, name):
    declare_components()

    with pytest.raises(ValueError, match=rf"'{group}\.{name}' uses the reserved name '{name}'"):
        TrainingSession(_config(tmp_path, {group: {name: {}}}))


@pytest.mark.parametrize("name", ["resources", "hooks", "steps", "role_bindings"])
def test_a_component_cannot_be_registered_under_a_reserved_name(name):
    class Reserved(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    with pytest.raises(ValueError, match=f"Cannot register a component named '{name}': it is a reserved configuration key"):
        resource(name)(Reserved)


# -- engine -----------------------------------------------------------------------


def test_the_configurator_finds_grouped_component_configs(tmp_path, monkeypatch):
    configurator = configurator_for(tmp_path, monkeypatch, {
        **_config(tmp_path, {}),
        "hooks": {"logger": {"log_every": 2}},
        "steps": {"compute#loss": {
            "dependencies_role_bindings": {"x": "y"},
            "outputs": "loss",
        }},
    })

    assert configurator.get_component_config(0, "logger") == {"log_every": 2}
    assert configurator.get_all_component_configs(0) == {
        "logger": {"log_every": 2},
        "compute#loss": {"outputs": "loss"},
    }
    with pytest.raises(KeyError):
        configurator.get_component_config(0, "steps")


def test_a_command_line_override_may_address_a_group(tmp_path, monkeypatch):
    path = tmp_path / "configurator.yaml"
    path.write_text(yaml.safe_dump({"sessions": [{
        **_config(tmp_path, {}),
        "hooks": {"logger": {"log_every": 2}},
    }]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--config", str(path),
        "--override", "sessions[0].hooks.logger.log_every=7",
    ])

    assert Configurator().get_component_config(0, "logger") == {"log_every": 7}


def test_the_engine_reads_ddp_listed_under_resources(tmp_path):
    engine = TrainingEngine(SimpleNamespace(
        process_timeout_on_join=10,
        heartbeat_timeout=30,
        topology_overrides=None,
    ))

    # Found, so its missing world_size is reported rather than ignored.
    with pytest.raises(ValueError, match="world_size"):
        engine.register_session(_config(tmp_path, {
            "resources": {"ddp": {"backend": "gloo"}},
        }))


def test_a_worker_records_the_launch_topology_where_ddp_is_listed(tmp_path):
    register_test_components()
    state = TrainingSession({
        "session_config": {
            "rng_seed": 7,
            "sessions_dir": str(tmp_path),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "role_bindings": {"model": "it_3d45_model"},
        "resources": {
            "ddp": {
                "world_size": 8,
                "backend": "gloo",
                "master_addr": "127.0.0.1",
                "master_port": "29500",
            },
            "it_3d45_model": {},
        },
    }).get_state()
    topology = LaunchTopology(
        world_size=4,
        backend="gloo",
        master_addr="10.0.0.1",
        master_port="29777",
        devices_per_node=0,
    )

    session = load_session_for_worker(state, 0, launch_topology=topology)

    assert session.full_config["resources"]["ddp"]["world_size"] == 4
    assert session.full_config["resources"]["ddp"]["master_port"] == "29777"
    assert "ddp" not in session.full_config
    assert state["config"]["resources"]["ddp"]["world_size"] == 8
