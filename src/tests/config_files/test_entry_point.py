"""Includes through the training entry point: `Configurator` (`--config`,
`--override`) and a `TrainingEngine` launch, which records the files a run
was built from."""

from __future__ import annotations

import sys

import pytest
import yaml

from tests.test_utils import COMPONENTS_PACKAGE, register_test_components, session_config
from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import TrainingSession


def _files(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(content), encoding="utf-8")


def _configurator(monkeypatch, config_path, *args):
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), *args])
    return Configurator()


def _training_files(tmp_path):
    """A session whose model settings come from a fragment."""
    session = session_config(tmp_path / "runs", max_iterations=1)
    model = session.pop("it_3d45_model")
    session["session_config"]["show_execution_graph"] = False
    _files(tmp_path, {
        "configs/train.yaml": {"sessions": [{"includes": ["common/model.yaml"], **session}]},
        "configs/common/model.yaml": {"it_3d45_model": model},
    })
    return tmp_path / "configs" / "train.yaml", model


def test_the_configurator_merges_includes_then_applies_overrides(tmp_path, monkeypatch):
    _files(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["base.yaml"], "model": {"depth": 4}}]},
        "base.yaml": {"model": {"depth": 2, "width": 8}},
    })

    configurator = _configurator(monkeypatch, tmp_path / "train.yaml", "--override", "sessions[0].model.width=16")

    assert configurator.session_configs == [{"model": {"depth": 4, "width": 16}}]
    (sources,) = configurator.config_sources
    assert [source.path.rsplit("/", 1)[-1] for source in sources] == ["train.yaml", "base.yaml"]


def test_a_config_without_includes_loads_as_before(tmp_path, monkeypatch):
    _files(tmp_path, {"train.yaml": {"sessions": [{"model": {"lr": 1e-3, "name": "x"}}]}})

    configurator = _configurator(monkeypatch, tmp_path / "train.yaml")

    assert configurator.session_configs == [{"model": {"lr": 1e-3, "name": "x"}}]


def test_a_bad_include_is_a_command_line_error(tmp_path, monkeypatch, capsys):
    _files(tmp_path, {"train.yaml": {"sessions": [{"includes": ["missing.yaml"]}]}})

    with pytest.raises(SystemExit) as raised:
        _configurator(monkeypatch, tmp_path / "train.yaml")

    assert raised.value.code == 2
    assert "missing.yaml: no such file" in capsys.readouterr().err


def test_an_interpolation_given_as_an_override_is_refused(tmp_path, monkeypatch, capsys):
    _files(tmp_path, {"train.yaml": {"sessions": [{"model": {"lr": 0.1}}]}})

    with pytest.raises(SystemExit):
        _configurator(monkeypatch, tmp_path / "train.yaml", "--override", "sessions[0].model.lr=${oc.env:LR}")

    assert "interpolation (${...}) is not supported" in capsys.readouterr().err


def test_an_interpolation_in_an_extension_override_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "train", "--extend-session", str(tmp_path / "ckpt"), "--override", "model.lr=${x}",
    ])

    with pytest.raises(SystemExit):
        Configurator()

    assert "interpolation (${...}) is not supported" in capsys.readouterr().err


def test_an_escaped_dollar_brace_reaches_the_session_as_text(tmp_path, monkeypatch):
    (tmp_path / "train.yaml").write_text("sessions:\n  - note: \\${literal}\n", encoding="utf-8")

    configurator = _configurator(monkeypatch, tmp_path / "train.yaml")

    assert configurator.session_configs == [{"note": "${literal}"}]


def test_a_session_given_includes_directly_is_refused(tmp_path):
    register_test_components()
    config = session_config(tmp_path, max_iterations=1)
    config["includes"] = ["common/model.yaml"]

    with pytest.raises(ValueError, match="still lists 'includes'"):
        TrainingSession(config)


def test_a_launch_records_its_config_files_and_a_complete_config(tmp_path, monkeypatch):
    register_test_components()
    config_path, model = _training_files(tmp_path)
    configurator = _configurator(
        monkeypatch, config_path, "--heartbeat-timeout", "60", "--process_timeout_on_join", "10",
    )

    with TrainingEngine(configurator) as engine:
        engine.start_session()

    (session_dir,) = (tmp_path / "runs").glob("session_*")
    written = yaml.safe_load((session_dir / "config.yaml").read_text())
    assert "includes" not in written
    assert written["it_3d45_model"] == model
    assert written["session_config"]["components_package"] == COMPONENTS_PACKAGE
    sources = yaml.safe_load((session_dir / "config_sources.yaml").read_text())["sources"]
    assert [(s["path"], s["included_by"]) for s in sources] == [
        (str(config_path.resolve()), None),
        (str((tmp_path / "configs" / "common" / "model.yaml").resolve()), str(config_path.resolve())),
    ]
    assert all(len(s["sha256"]) == 64 for s in sources)
