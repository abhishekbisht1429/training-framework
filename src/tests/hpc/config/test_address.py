import pytest

from training_framework.hpc.config import address

SESSION = {
    "session_config": {"rng_seed": 0, "sessions_dir": "./runs"},
    "resources": {"optimizer": {"optimizer": {"kwargs": {"lr": 0.1}}}},
    "flat.name": {"value": 1},
    "empty": None,
}


def parse(raw):
    parsed, problems = address.parse_address(raw, "where")
    assert not problems, problems
    return parsed


def test_a_component_is_found_at_the_top_level_and_in_a_group():
    grouped = parse({"component": "optimizer", "path": "optimizer.kwargs.lr"})
    flat = parse({"component": "flat.name", "path": "value"})
    assert address.read(SESSION, grouped) == 0.1
    assert address.read(SESSION, flat) == 1


def test_writes_keep_types_and_create_missing_keys():
    target = parse({"component": "optimizer", "path": ["optimizer", "betas"]})
    updated = address.write(SESSION, target, [0.9, 0.99])
    assert updated["resources"]["optimizer"]["optimizer"]["betas"] == [0.9, 0.99]
    assert SESSION["resources"]["optimizer"]["optimizer"] == {"kwargs": {"lr": 0.1}}


def test_list_paths_address_keys_containing_dots():
    session = {"m": {"layer.0": {"w": 1}}}
    session_target = parse({"path": ["m", "layer.0", "w"]})
    assert address.read(session, session_target) == 1
    assert address.write(session, session_target, 2)["m"]["layer.0"]["w"] == 2


def test_an_empty_entry_reads_as_an_empty_mapping():
    target = parse({"component": "empty", "path": "x"})
    assert address.is_missing(address.read(SESSION, target))
    assert address.write(SESSION, target, 3)["empty"] == {"x": 3}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"component": "missing", "path": "x"}, "not listed"),
        ({"component": "resources", "path": "optimizer"}, "component group"),
        ({"path": "resources.optimizer.lr"}, "component group"),
        ({"path": "session_config.sessions_dir"}, "set by the HPC layer"),
        ({"path": "session_config"}, "set by the HPC layer"),
        ({"component": "session_config", "path": "rng_seed"}, "not listed"),
    ],
)
def test_bad_targets_are_refused(raw, message):
    problem = address.check_target(parse(raw), SESSION)
    assert problem is not None and message in problem


@pytest.mark.parametrize("path", ["", [], ["a", ""], "a..b", ["a[0]"], 3])
def test_malformed_paths_are_refused(path):
    parsed, problems = address.parse_address({"path": path}, "where")
    assert parsed is None and problems
