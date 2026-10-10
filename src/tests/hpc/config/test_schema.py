import copy

import pytest

from training_framework.hpc.errors import PlanError
from training_framework.hpc.config.schema import parse_manifest, parse_profiles
from tests.hpc.batch_repo import MANIFEST, PROFILE


def test_a_complete_profile_and_manifest_parse():
    profile = parse_profiles(PROFILE, None, "p.yaml")
    assert profile.name == "cluster"
    assert profile.slurm.partitions == {"gpu": "gpu", "cpu": "general"}
    assert profile.env.site_python_identity.version == "3.12.4"
    assert profile.setup_resources.memory == "16G"
    manifest = parse_manifest(copy.deepcopy(MANIFEST), "m.yaml")
    assert manifest.groups[0].resources.gpus == 1
    assert manifest.groups[0].stop_grace_seconds == 120


def test_the_selected_profile_wins_over_the_default():
    data = copy.deepcopy(PROFILE)
    data["profiles"]["other"] = copy.deepcopy(data["profiles"]["cluster"])
    assert parse_profiles(data, "other", "p.yaml").name == "other"
    with pytest.raises(PlanError, match="profile 'nope' not found"):
        parse_profiles(data, "nope", "p.yaml")


@pytest.mark.parametrize("key", ["account", "path_map", "hooks", "profile", "ssh_host"])
def test_site_keys_are_refused_in_a_manifest(key):
    data = copy.deepcopy(MANIFEST)
    data[key] = "x"
    with pytest.raises(PlanError, match="belongs in the profile"):
        parse_manifest(data, "m.yaml")


@pytest.mark.parametrize("key", ["partition", "qos", "account"])
def test_site_placement_is_refused_in_portable_resources(key):
    data = copy.deepcopy(MANIFEST)
    data["groups"]["probes"]["resources"][key] = "x"
    with pytest.raises(PlanError, match="belongs in the profile"):
        parse_manifest(data, "m.yaml")


def test_every_problem_is_reported_at_once():
    data = copy.deepcopy(MANIFEST)
    data["name"] = "bad name"
    data["surprise"] = 1
    data["groups"]["probes"]["resources"]["walltime"] = "1h"
    data["groups"]["probes"]["experiments"][0]["matrix"][0]["values"] = []
    with pytest.raises(PlanError) as error:
        parse_manifest(data, "m.yaml")
    text = str(error.value)
    for fragment in ("m.yaml.name", "unknown field 'surprise'", "walltime", "non-empty list"):
        assert fragment in text


@pytest.mark.parametrize("name", ["", "a b", "a/b", "a.b", 3])
def test_unsafe_batch_names_are_refused(name):
    data = copy.deepcopy(MANIFEST)
    data["name"] = name
    with pytest.raises(PlanError, match="letters, digits"):
        parse_manifest(data, "m.yaml")


def test_a_numeric_python_pin_is_refused_with_a_hint():
    data = copy.deepcopy(MANIFEST)
    data["environment"]["python"] = 3.1
    with pytest.raises(PlanError, match="quoted string"):
        parse_manifest(data, "m.yaml")


def test_duplicate_experiment_names_are_refused():
    data = copy.deepcopy(MANIFEST)
    experiments = data["groups"]["probes"]["experiments"]
    experiments.append(copy.deepcopy(experiments[0]))
    with pytest.raises(PlanError, match="used twice"):
        parse_manifest(data, "m.yaml")
