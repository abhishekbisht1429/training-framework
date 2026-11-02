"""Includes in an HPC batch's training config: read at the batch's commit
(which must be the whole working tree), merged into every rendered task
config."""

from __future__ import annotations

import copy

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, PROFILE, TRAINING_CONFIG, make_repo
from training_framework.hpc.__main__ import main

OPTIMIZER = {"optimizer": {"optimizer": {"name": "AdamW", "kwargs": {"lr": 0.001, "weight_decay": 0.05}}}}


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "hpc.yaml"
    path.write_text(yaml.safe_dump(PROFILE))
    return path


def _repo(tmp_path, includes, fragments):
    config = copy.deepcopy(TRAINING_CONFIG)
    session = config["sessions"][0]
    del session["resources"]
    session["includes"] = includes
    return make_repo(tmp_path, config=config, extra_files={
        name: yaml.safe_dump(content) for name, content in fragments.items()
    })


def _render(root, profiles, output):
    return main(["render", str(root / "batch.yaml"), "--profiles", str(profiles), "--output", str(output)])


def test_task_configs_are_rendered_with_the_includes_merged(tmp_path, profiles):
    root = _repo(tmp_path, ["common/optimizer.yaml"], {"configs/common/optimizer.yaml": {"resources": OPTIMIZER}})

    assert _render(root, profiles, tmp_path / "preview") == 0

    task = yaml.safe_load((tmp_path / "preview" / "tasks" / "probes" / "frozen" / "001" / "config.yaml").read_text())
    session = task["sessions"][0]
    assert "includes" not in session
    # The matrix sets lr on the included optimizer; the rest comes from the fragment.
    assert session["resources"]["optimizer"]["optimizer"]["kwargs"] == {"lr": 0.001, "weight_decay": 0.05}


def test_an_uncommitted_change_to_an_included_file_is_refused(tmp_path, profiles, capsys):
    # A batch runs one exact commit, includes as much as the config itself.
    root = _repo(tmp_path, ["common/optimizer.yaml"], {"configs/common/optimizer.yaml": {"resources": OPTIMIZER}})
    changed = copy.deepcopy(OPTIMIZER)
    changed["optimizer"]["optimizer"]["kwargs"]["weight_decay"] = 0.5
    (root / "configs" / "common" / "optimizer.yaml").write_text(yaml.safe_dump({"resources": changed}))

    assert _render(root, profiles, tmp_path / "preview") == 2

    assert "uncommitted changes" in capsys.readouterr().err


def test_an_include_leaving_the_repository_fails_validation(tmp_path, profiles, capsys):
    root = _repo(tmp_path, ["../../elsewhere.yaml"], {})

    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(profiles)]) == 2

    assert "leaves the repository" in capsys.readouterr().err


def test_a_missing_include_fails_validation(tmp_path, profiles, capsys):
    root = _repo(tmp_path, ["common/missing.yaml"], {})

    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(profiles)]) == 2

    assert "configs/common/missing.yaml: no such file" in capsys.readouterr().err


def test_an_interpolation_brought_in_by_a_matrix_value_fails_validation(tmp_path, profiles, capsys):
    manifest = copy.deepcopy(MANIFEST)
    manifest["groups"]["probes"]["experiments"][0]["matrix"][0]["values"] = ["${oc.env:LR}"]
    root = make_repo(tmp_path, manifest=manifest)

    assert main(["validate", str(root / "batch.yaml"), "--profiles", str(profiles)]) == 2

    assert "interpolation (${...}) is not supported" in capsys.readouterr().err
