"""Tests for `dependencies_role_bindings` declared inside a component's own entry.

The entry wires that component alone: it means what a nested entry of the
deprecated top-level `component_bindings` meant, and the component's
constructor never sees it.
"""

from dataclasses import dataclass

import pytest

from training_framework.components import (
    ComponentDependencyError,
    ExtendableComponent,
    Resource,
    Step,
    parse_component_config,
    requires_resource,
    resource,
    step,
)
from training_framework.components.builtin import Checkpointer
from training_framework.session import TrainingSession


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
    """Register `crb_dep` and two steps requiring it: `crb_consumer`, which
    records the configuration it was built with and accepts extension, and
    `crb_other`."""

    @resource("crb_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("crb_dep")
    @step("crb_consumer")
    class Consumer(Step, ExtendableComponent):
        def __init__(self, config=None):
            super().__init__(config)
            self.built_with = dict(config) if config is not None else None
            self.extended_with = None

        def run(self, session) -> None:
            pass

        def apply_extension_config(self, config, changed_paths) -> None:
            self.extended_with = dict(config)

    @requires_resource("crb_dep")
    @step("crb_other")
    class Other(Step):
        def run(self, session) -> None:
            pass


def _step(session, name):
    (found,) = [
        component for component in session.get_all_steps()
        if component.name == name
    ]
    return found


def _given(session, consumer):
    return _step(session, consumer).get_dependency("crb_dep").name


TWO_DEPENDENCIES = {"crb_dep#a": {}, "crb_dep#b": {}}


def test_a_component_names_the_instance_it_is_given(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {
            "dependencies_role_bindings": {"crb_dep": "crb_dep#b"},
            "every": 3,
        },
    }))

    assert _given(session, "crb_consumer") == "crb_dep#b"
    assert "crb_consumer: crb_dep -> crb_dep#b" in session.execution_graph()


def test_the_constructor_never_sees_dependencies_role_bindings(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {
            "dependencies_role_bindings": {"crb_dep": "crb_dep#b"},
            "every": 3,
        },
    }))

    assert _step(session, "crb_consumer").built_with == {"every": 3}
    # The configuration is kept as written.
    assert session.full_config["crb_consumer"]["dependencies_role_bindings"] == {
        "crb_dep": "crb_dep#b",
    }


def test_a_config_schema_never_sees_dependencies_role_bindings(tmp_path):
    declare_components()

    @dataclass
    class ScheduleConfig:
        every: int = 1

    @requires_resource("crb_dep")
    @step("crb_scheduled")
    class Scheduled(Step):
        config_schema = ScheduleConfig

        def __init__(self, config=None):
            # The schema refuses keys it does not declare,
            # `dependencies_role_bindings` among them.
            super().__init__(config)
            self.every = parse_component_config(type(self), config).every

        def run(self, session) -> None:
            pass

    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_scheduled": {
            "dependencies_role_bindings": {"crb_dep": "crb_dep#a"},
            "every": 2,
        },
    }))

    scheduled = _step(session, "crb_scheduled")
    assert scheduled.every == 2
    assert scheduled.get_dependency("crb_dep").name == "crb_dep#a"


def test_own_wiring_wins_over_the_session_wide_binding_for_that_component(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        "role_bindings": {"crb_dep": "crb_dep#a"},
        **TWO_DEPENDENCIES,
        "crb_consumer": {"dependencies_role_bindings": {"crb_dep": "crb_dep#b"}},
        "crb_other": {},
    }))

    assert _given(session, "crb_consumer") == "crb_dep#b"
    assert _given(session, "crb_other") == "crb_dep#a"


def test_an_instance_entry_may_hold_its_own_wiring(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer#x": {"dependencies_role_bindings": {"crb_dep": "crb_dep#a"}},
        "crb_consumer#y": {"dependencies_role_bindings": {"crb_dep": "crb_dep#b"}},
    }))

    assert _given(session, "crb_consumer#x") == "crb_dep#a"
    assert _given(session, "crb_consumer#y") == "crb_dep#b"


def test_an_entry_holding_only_its_wiring_configures_the_component(tmp_path):
    declare_components()

    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {"dependencies_role_bindings": {"crb_dep": "crb_dep#a"}},
    }))

    assert _step(session, "crb_consumer").built_with == {}


def test_wiring_a_component_in_both_places_is_refused(tmp_path):
    declare_components()

    with (
        pytest.warns(DeprecationWarning, match="component_bindings"),
        pytest.raises(ValueError, match="wired both by its own dependencies_"),
    ):
        TrainingSession(_config(tmp_path, {
            "component_bindings": {"crb_consumer": {"crb_dep": "crb_dep#a"}},
            **TWO_DEPENDENCIES,
            "crb_consumer": {"dependencies_role_bindings": {"crb_dep": "crb_dep#b"}},
        }))


@pytest.mark.parametrize(
    ("wiring", "error", "match"),
    (
        pytest.param(["crb_dep"], TypeError, "must be a mapping", id="not-a-mapping"),
        pytest.param(
            {"crb_dep": {"x": "crb_dep#a"}},
            ValueError,
            "binds 'crb_dep' to a mapping",
            id="nested",
        ),
        pytest.param({"crb_dep": 1}, TypeError, "strings to strings", id="non-string"),
        pytest.param(
            {"crb_dep": "nothing_registered"},
            ValueError,
            "not a registered component",
            id="unregistered",
        ),
        pytest.param(
            {"crb_dep#a": "crb_dep"},
            ValueError,
            "role name",
            id="suffixed-role",
        ),
        pytest.param(
            {"crb_dep": "crb_dep#c"},
            ComponentDependencyError,
            "not configured",
            id="unconfigured-instance",
        ),
    ),
)
def test_malformed_wiring_is_refused(tmp_path, wiring, error, match):
    declare_components()

    with pytest.raises(error, match=match):
        TrainingSession(_config(tmp_path, {
            **TWO_DEPENDENCIES,
            "crb_consumer": {"dependencies_role_bindings": wiring},
        }))


def test_wiring_survives_a_state_round_trip(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {"dependencies_role_bindings": {"crb_dep": "crb_dep#b"}},
    }))

    restored = TrainingSession.from_state(session.get_state())

    assert _given(restored, "crb_consumer") == "crb_dep#b"
    assert _step(restored, "crb_consumer").built_with == {}


def test_wiring_survives_a_checkpoint_on_disk(tmp_path):
    declare_components()

    @requires_resource("crb_dep")
    @resource("crb_holder")
    class Holder(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {"dependencies_role_bindings": {"crb_dep": "crb_dep#b"}},
        "crb_holder": {"dependencies_role_bindings": {"crb_dep": "crb_dep#a"}},
    }))
    saved = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")

    loaded = Checkpointer.load_checkpoint(saved)
    holder = Checkpointer.load_component(saved, "crb_holder")

    assert _given(loaded, "crb_consumer") == "crb_dep#b"
    assert holder.get_dependency("crb_dep").name == "crb_dep#a"


def test_extension_refuses_to_change_a_components_wiring(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {"dependencies_role_bindings": {"crb_dep": "crb_dep#b"}},
    }))

    with pytest.raises(
            ValueError,
            match=(
                r"does not allow changes to: "
                r"crb_consumer\.dependencies_role_bindings\.crb_dep"
            ),
    ):
        session.apply_extension_overrides(
            ("crb_consumer.dependencies_role_bindings.crb_dep=crb_dep#a",)
        )


def test_extension_passes_the_component_its_config_without_wiring(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        **TWO_DEPENDENCIES,
        "crb_consumer": {
            "dependencies_role_bindings": {"crb_dep": "crb_dep#b"},
            "every": 1,
        },
    }))

    session.apply_extension_overrides(("crb_consumer.every=5",))

    assert _step(session, "crb_consumer").extended_with == {"every": 5}
    assert session.full_config["crb_consumer"] == {
        "dependencies_role_bindings": {"crb_dep": "crb_dep#b"},
        "every": 5,
    }


@pytest.mark.parametrize("entry", [
    {},
    {"dependencies_role_bindings": {"crb_dep": "crb_dep"}},
])
def test_a_bound_role_configured_as_a_component_is_refused_with_or_without_its_own_wiring(tmp_path, entry):
    declare_components()
    # `crb_consumer` is a session-wide role here, so configuring it as a
    # component is refused -- also when its entry carries its own wiring,
    # which must not replace the session-wide binding of the same name.
    config = _config(tmp_path, {
        "role_bindings": {"crb_consumer": "crb_other"},
        "crb_dep": {},
        "crb_consumer": entry,
    })

    with pytest.raises(ValueError, match="Component role 'crb_consumer' is bound to 'crb_other'. Configure the implementation name 'crb_other'"):
        TrainingSession(config)
