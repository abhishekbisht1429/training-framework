"""Compatibility tests: `hpc/framework.py` wraps internal session-config
helpers; these pin the behaviour the HPC layer relies on."""

from training_framework.hpc import framework

SESSION = {
    "session_config": {"rng_seed": 0},
    "role_bindings": {"loss": "compute"},
    "resources": {"ddp": {"world_size": 2}, "empty": None},
    "logger": {"log_every": 1},
}


def test_listed_components_are_found_flat_and_grouped():
    assert framework.is_listed_component(SESSION, "ddp")
    assert framework.is_listed_component(SESSION, "logger")
    assert not framework.is_listed_component(SESSION, "missing")


def test_reserved_names_are_not_components():
    assert not framework.is_listed_component(SESSION, "session_config")
    assert not framework.is_listed_component(SESSION, "role_bindings")
    assert {"resources", "hooks", "steps"} <= set(framework.COMPONENT_GROUPS)


def test_entries_are_read_and_replaced_where_they_are_listed():
    assert framework.component_entry_of(SESSION, "ddp") == {"world_size": 2}
    assert framework.component_entry_of(SESSION, "empty") == {}
    updated = framework.with_entry(SESSION, "ddp", {"world_size": 4})
    assert updated["resources"]["ddp"] == {"world_size": 4}
    assert "ddp" not in updated
    assert SESSION["resources"]["ddp"] == {"world_size": 2}
