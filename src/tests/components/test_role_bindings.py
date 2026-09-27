"""Tests for session-wide `role_bindings`, its deprecated names, and the
`RoleBindings` class and keyword that replaced `ComponentBindings`.

Bindings declared inside a component's own entry are covered in
`test_component_role_bindings.py`.
"""

import pickle
import warnings

import pytest

import training_framework.components as components
from training_framework.components import (
    ComponentAliases,
    Resource,
    RoleBindings,
    Step,
    format_execution_graph,
    requires_resource,
    resource,
    step,
    topological_sort_of_components,
)
from training_framework.session import SessionComponents, TrainingSession


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
    """Register `rb_impl`, and `rb_consumer` requiring the role `rb_role`."""

    @resource("rb_impl")
    class Implementation(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("rb_role")
    @step("rb_consumer")
    class Consumer(Step):
        def run(self, session) -> None:
            pass


def _deprecations(caught):
    return [
        str(warning.message) for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


def _consumer(session):
    (consumer,) = [
        component for component in session.get_all_steps()
        if component.name == "rb_consumer"
    ]
    return consumer


def test_top_level_role_bindings_bind_a_role_for_the_whole_session(tmp_path):
    declare_components()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = TrainingSession(_config(tmp_path, {
            "role_bindings": {"rb_role": "rb_impl"},
            "rb_consumer": {},
        }))

    assert _deprecations(caught) == []
    assert session.resolve_component_name("rb_role") == "rb_impl"
    assert _consumer(session).get_dependency("rb_role").name == "rb_impl"
    assert "ROLE BINDINGS" in session.execution_graph()
    assert "rb_role -> rb_impl" in session.execution_graph()


def test_top_level_role_bindings_refuse_per_component_wiring(tmp_path):
    declare_components()

    with pytest.raises(ValueError, match=r"rb_consumer: \{dependencies_role_bindings"):
        TrainingSession(_config(tmp_path, {
            "role_bindings": {"rb_consumer": {"rb_role": "rb_impl"}},
            "rb_consumer": {},
        }))


def test_top_level_role_bindings_must_be_a_mapping(tmp_path):
    declare_components()

    with pytest.raises(TypeError, match="'role_bindings' must be a mapping"):
        TrainingSession(_config(tmp_path, {"role_bindings": ["rb_role"]}))


@pytest.mark.parametrize(
    "keys",
    (
        ("role_bindings", "component_bindings"),
        ("role_bindings", "aliases"),
        ("component_bindings", "aliases"),
    ),
)
def test_only_one_top_level_bindings_entry_may_be_configured(tmp_path, keys):
    declare_components()

    with pytest.raises(ValueError, match="not both"):
        TrainingSession(_config(tmp_path, {key: {} for key in keys}))


def test_legacy_component_bindings_still_bind_and_wire_but_warn(tmp_path):
    declare_components()

    with pytest.warns(DeprecationWarning, match="use 'role_bindings'"):
        session = TrainingSession(_config(tmp_path, {
            "component_bindings": {
                "rb_role": "rb_impl#a",
                "rb_consumer": {"rb_role": "rb_impl#b"},
            },
            "rb_impl#a": {},
            "rb_impl#b": {},
            "rb_consumer": {},
        }))

    assert session.resolve_component_name("rb_role") == "rb_impl#a"
    assert _consumer(session).get_dependency("rb_role").name == "rb_impl#b"


def test_restoring_a_legacy_configuration_does_not_warn(tmp_path):
    declare_components()
    with pytest.warns(DeprecationWarning):
        state = TrainingSession(_config(tmp_path, {
            "component_bindings": {"rb_role": "rb_impl"},
            "rb_consumer": {},
        })).get_state()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = TrainingSession.from_state(state)

    assert _deprecations(caught) == []
    assert restored.resolve_component_name("rb_role") == "rb_impl"


def test_session_wide_role_bindings_survive_a_state_round_trip(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        "role_bindings": {"rb_role": "rb_impl"},
        "rb_consumer": {},
    }))

    restored = TrainingSession.from_state(session.get_state())

    assert restored.resolve_component_name("rb_role") == "rb_impl"
    assert _consumer(restored).get_dependency("rb_role").name == "rb_impl"


def test_extension_refuses_to_change_top_level_role_bindings(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        "role_bindings": {"rb_role": "rb_impl"},
        "rb_consumer": {},
    }))

    with pytest.raises(ValueError, match="does not allow changes to: role_bindings"):
        session.apply_extension_overrides(("role_bindings.rb_role=rb_other",))


# -- RoleBindings and the role_bindings keyword -------------------------------


def test_component_bindings_is_a_deprecated_name_for_role_bindings():
    with pytest.warns(DeprecationWarning, match="use RoleBindings"):
        legacy = components.ComponentBindings

    assert legacy is RoleBindings
    assert "RoleBindings" in components.__all__
    assert "ComponentBindings" not in components.__all__


def test_component_aliases_is_still_a_role_bindings():
    declare_components()

    with pytest.warns(DeprecationWarning, match="use RoleBindings"):
        legacy = ComponentAliases({"rb_role": "rb_impl"})

    assert isinstance(legacy, RoleBindings)


def test_bindings_pickled_under_the_former_class_name_load_silently():
    """Checkpoints written before the rename name `ComponentBindings`."""
    bindings = RoleBindings({"log_role": "logger"})
    # Protocol 0 names a class as text, so the old name can be put back.
    old_pickle = pickle.dumps(bindings, protocol=0).replace(
        b"\nRoleBindings\n",
        b"\nComponentBindings\n",
    )
    assert b"ComponentBindings" in old_pickle

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pickle.loads(old_pickle)

    assert _deprecations(caught) == []
    assert type(restored) is RoleBindings
    assert restored.resolve("log_role") == "logger"


def test_session_components_take_role_bindings():
    declare_components()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held = SessionComponents(role_bindings={"rb_role": "rb_impl"})

    assert _deprecations(caught) == []
    assert held.resolve_name("rb_role") == "rb_impl"


def test_session_components_still_take_component_bindings_with_a_warning():
    declare_components()

    with pytest.warns(DeprecationWarning, match="use 'role_bindings'"):
        held = SessionComponents(component_bindings={"rb_role": "rb_impl"})

    assert held.resolve_name("rb_role") == "rb_impl"


@pytest.mark.parametrize(
    "keywords",
    (
        {"role_bindings": {}, "component_bindings": {}},
        {"role_bindings": {}, "aliases": {}},
        {"component_bindings": {}, "aliases": {}},
    ),
)
def test_only_one_bindings_keyword_may_be_given(keywords):
    with pytest.raises(ValueError, match="not both"):
        SessionComponents(**keywords)


def test_graph_functions_take_role_bindings(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        "role_bindings": {"rb_role": "rb_impl"},
        "rb_consumer": {},
    }))
    listed = (
        session.get_all_resources()
        + session.get_all_hooks()
        + session.get_all_steps()
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        order = topological_sort_of_components(
            {"rb_role": "rb_impl"},
            components=listed,
        )
        graph = format_execution_graph(
            resources=session.get_all_resources(),
            hooks=session.get_all_hooks(),
            steps=session.get_all_steps(),
            max_iterations=1,
            role_bindings={"rb_role": "rb_impl"},
        )

    implementation = next(
        component for component in listed if component.name == "rb_impl"
    )
    assert _deprecations(caught) == []
    assert order[implementation.id] < order[_consumer(session).id]
    assert "rb_role -> rb_impl" in graph

    with pytest.warns(DeprecationWarning, match="use 'role_bindings'"):
        format_execution_graph(
            resources=session.get_all_resources(),
            hooks=session.get_all_hooks(),
            steps=session.get_all_steps(),
            max_iterations=1,
            component_bindings={"rb_role": "rb_impl"},
        )


# -- session-level names -------------------------------------------------------


def test_the_session_exposes_its_role_bindings(tmp_path):
    declare_components()
    session = TrainingSession(_config(tmp_path, {
        "role_bindings": {"rb_role": "rb_impl"},
        "rb_consumer": {},
    }))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bindings = session.role_bindings

    assert _deprecations(caught) == []
    assert bindings == {"rb_role": "rb_impl"}
    with pytest.warns(DeprecationWarning, match="use role_bindings"):
        assert session.component_bindings == bindings


def test_session_components_name_their_bindings_role_bindings():
    declare_components()
    held = SessionComponents(role_bindings={"rb_role": "rb_impl"})

    assert isinstance(held.role_bindings, RoleBindings)
    with pytest.warns(DeprecationWarning, match="use role_bindings"):
        assert held.component_bindings is held.role_bindings


def test_session_components_pickled_under_the_former_attribute_name_restore():
    declare_components()
    held = SessionComponents(role_bindings={"rb_role": "rb_impl"})
    state = held.__dict__.copy()
    state["component_bindings"] = state.pop("role_bindings")

    restored = SessionComponents.__new__(SessionComponents)
    restored.__setstate__(state)

    assert restored.resolve_name("rb_role") == "rb_impl"
    assert "component_bindings" not in restored.__dict__
