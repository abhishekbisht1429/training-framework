import json

import pytest
import yaml

from training_framework.hpc.__main__ import main
from tests.hpc.batch_repo import PROFILE, TRAINING_CONFIG, make_repo


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    profiles = tmp_path / "hpc.yaml"
    profiles.write_text(yaml.safe_dump(PROFILE))
    return make_repo(tmp_path), profiles


def test_validate_reports_the_plan(setup, capsys):
    root, profiles = setup
    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(profiles)]) == 0
    assert "ok: 4 task(s) in 1 group(s)" in capsys.readouterr().out


def test_validate_reports_every_problem_and_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    profiles = tmp_path / "hpc.yaml"
    profiles.write_text(yaml.safe_dump(PROFILE))
    root = make_repo(tmp_path, manifest={"version": 2, "name": "x y"})
    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(profiles)]) == 2
    error = capsys.readouterr().err
    assert "version must be 1" in error and "letters, digits" in error and "problems" in error


def test_suggest_data_prints_entries_for_local_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    profiles = tmp_path / "hpc.yaml"
    profiles.write_text(yaml.safe_dump(PROFILE))
    data = tmp_path / "datasets" / "flowers"
    data.mkdir(parents=True)
    config = json.loads(json.dumps(TRAINING_CONFIG))
    config["sessions"][0]["flowers102"]["root"] = str(data)
    root = make_repo(tmp_path, config=config)
    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(profiles),
                 "--suggest-data"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out.split("\n", 2)[-1])
    assert printed["data"]["sources"][0]["target"] == {"component": "flowers102", "path": ["root"]}


def test_render_writes_the_preview_and_refuses_a_non_empty_directory(setup, tmp_path, capsys):
    root, profiles = setup
    output = tmp_path / "preview"
    args = ["render", str(root / "batch.yaml"), "--profiles", str(profiles), "--output", str(output)]
    assert main(args) == 0
    assert (output / "plan.json").exists()
    assert (output / "groups" / "probes" / "tasks.tsv").read_text().count("\n") == 5
    assert (output / "tasks" / "probes" / "frozen" / "003" / "config.yaml").exists()
    assert (output / "nodes" / "setup" / "setup.sbatch").exists()
    assert main(args) == 2
    assert "is not empty" in capsys.readouterr().err


def test_a_missing_profile_file_is_an_error(setup, tmp_path, capsys):
    root, _ = setup
    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(tmp_path / "none.yaml")]) == 2
    assert "does not exist" in capsys.readouterr().err
