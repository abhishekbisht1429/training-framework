"""`python -m training_framework.reporting summarize`."""

from __future__ import annotations

import json
import subprocess
import sys

from training_framework.reporting import DEFAULT_IGNORED_KEYS
from training_framework.reporting.__main__ import main

from .helpers import SESSION, config, losses, write_run


def _runs(root):
    for lr in (0.5, 0.25):
        for seed in (1, 2):
            write_run(root / f"lr{lr}_s{seed}", losses(2.0, lr + seed / 8), config(lr=lr, seed=seed))


def test_markdown_to_standard_output(tmp_path, capsys):
    _runs(tmp_path)

    assert main(["summarize", str(tmp_path), "--metric", "loss:min", "--sort", "loss"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("# Run summary")
    assert "## Seed groups" in out and "Sorted by: best `loss` (min)" in out


def test_output_file_and_formats(tmp_path, capsys):
    _runs(tmp_path / "runs")
    report = tmp_path / "reports" / "summary.json"

    assert main(["summarize", str(tmp_path / "runs"), "--format", "json", "-o", str(report)]) == 0
    assert main(["summarize", str(tmp_path / "runs"), "--format", "csv", "--table", "runs",
                 "-o", str(tmp_path / "runs.csv")]) == 0

    assert len(json.loads(report.read_text())["runs"]) == 4
    assert (tmp_path / "runs.csv").read_text().count("\n") == 5
    assert "wrote" in capsys.readouterr().err


def test_problems_fail_only_with_strict(tmp_path, capsys):
    write_run(tmp_path / "s1", losses(2.0, 1.0), config(seed=1))
    write_run(tmp_path / "s2", losses(2.0), config(seed=2))

    assert main(["summarize", str(tmp_path)]) == 0
    assert main(["summarize", str(tmp_path), "--strict"]) == 1

    assert "ends at different steps" in capsys.readouterr().err


def test_errors_exit_with_status_2(tmp_path, capsys):
    _runs(tmp_path)

    assert main(["summarize", str(tmp_path), "--metric", "accuracy"]) == 2
    assert main(["summarize", str(tmp_path), "--table", "runs"]) == 2
    assert main(["summarize", str(tmp_path), "--no-seed-groups", "--seed-key", "x"]) == 2
    assert main(["summarize", str(tmp_path), "--seed-key", "data_manager.seed"]) == 2

    err = capsys.readouterr().err
    assert "error: no run writes the tag 'accuracy'" in err
    assert "--table only applies with --format csv" in err
    assert "--seed-key has no effect with --no-seed-groups" in err


def test_the_report_is_not_written_into_a_run(tmp_path, capsys):
    _runs(tmp_path)
    inside = tmp_path / "lr0.5_s1" / SESSION / "summary.md"

    assert main(["summarize", str(tmp_path), "-o", str(inside)]) == 2

    assert not inside.exists()
    assert "is inside run" in capsys.readouterr().err


def test_ignore_keys_add_to_the_defaults_unless_cleared(tmp_path, capsys):
    write_run(tmp_path / "a", losses(1.0), config(sessions_dir="a", note="x"))
    write_run(tmp_path / "b", losses(1.0), config(sessions_dir="b", note="y"))

    main(["summarize", str(tmp_path), "--format", "json", "--ignore-key", "note"])
    added = json.loads(capsys.readouterr().out)
    main(["summarize", str(tmp_path), "--format", "json", "--no-default-ignores"])
    cleared = json.loads(capsys.readouterr().out)

    assert added["ignored_keys"] == [*DEFAULT_IGNORED_KEYS, "note"]
    assert added["columns"] == []
    assert [c["name"] for c in cleared["columns"]] == ["note", "session_config.sessions_dir"]


def test_runs_as_a_module(tmp_path):
    _runs(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "training_framework.reporting", "summarize", str(tmp_path),
         "--no-seed-groups", "--format", "csv"],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0].startswith("run,")
