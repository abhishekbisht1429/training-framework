"""Finding runs, reading their scalars and configs, and comparing them.

Values written are exact in float32 (TensorBoard stores scalars as float32),
so they compare exactly after the round trip.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics

import pytest

from training_framework.reporting import (
    DEFAULT_IGNORED_KEYS,
    Metric,
    ReportError,
    discover_runs,
    render_csv,
    render_json,
    render_markdown,
    summarize_runs,
)

from .helpers import SESSION, WRITER, config, losses, write_attempt, write_run


def _summary(*roots, metrics=("loss:min",), **options):
    return summarize_runs(discover_runs(*roots), metrics, **options)


def _by_id(summary):
    return {run.run_id: run for run in summary.runs}


def _lr_runs(root, seeds=(1,), lrs=(0.5, 0.25)):
    for lr in lrs:
        for seed in seeds:
            write_run(
                root / f"lr{lr}_seed{seed}", losses(4.0, 3.0 + lr, 1.0 + lr + seed / 8),
                config(lr=lr, seed=seed, sessions_dir=str(root / f"lr{lr}_seed{seed}")),
            )


# -- discovery -------------------------------------------------------------------


def test_each_session_directory_is_a_run_named_by_its_path(tmp_path):
    write_run(tmp_path / "exp_a", losses(1.0), config())
    write_run(tmp_path / "exp_b" / "nested", losses(1.0), config())

    runs = discover_runs(tmp_path).runs

    assert [run.run_id for run in runs] == [
        f"exp_a/{SESSION}", f"exp_b/nested/{SESSION}",
    ]
    assert runs[0].writers == [WRITER]
    assert runs[0].task is None and runs[0].variant == {}


def test_a_session_directory_given_as_the_root_is_one_run(tmp_path):
    session_dir = write_run(tmp_path, losses(1.0), config())

    (run,) = discover_runs(session_dir).runs

    assert run.run_id == SESSION


def test_two_roots_prefix_run_ids_with_the_root_name(tmp_path):
    write_run(tmp_path / "local" / "baseline", losses(1.0), config())
    write_run(tmp_path / "fetched" / "baseline", losses(1.0), config())

    runs = discover_runs(tmp_path / "local", tmp_path / "fetched").runs

    assert sorted(run.run_id for run in runs) == [
        f"fetched/baseline/{SESSION}", f"local/baseline/{SESSION}",
    ]


def test_two_roots_whose_runs_get_the_same_id_are_refused(tmp_path):
    write_run(tmp_path / "a" / "runs" / "x", losses(1.0), config())
    write_run(tmp_path / "b" / "runs" / "x", losses(1.0), config())

    with pytest.raises(ReportError, match="both be called 'runs/x/session_"):
        discover_runs(tmp_path / "a" / "runs", tmp_path / "b" / "runs")


def test_nested_roots_are_refused(tmp_path):
    write_run(tmp_path / "x", losses(1.0), config())

    with pytest.raises(ReportError, match="under two of the given directories"):
        discover_runs(tmp_path, tmp_path / "x")


def test_a_root_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(ReportError, match="is not a directory"):
        discover_runs(tmp_path / "missing")


def test_include_and_exclude_filter_run_ids(tmp_path):
    for name in ("sgd_a", "sgd_b", "adam_a"):
        write_run(tmp_path / name, losses(1.0), config())

    included = discover_runs(tmp_path, include=["sgd_*"], exclude=["*_b/*"]).runs

    assert [run.run_id for run in included] == [f"sgd_a/{SESSION}"]


def test_event_files_outside_a_session_directory_are_reported(tmp_path):
    write_run(tmp_path / "exp", losses(1.0), config())
    stray = tmp_path / "other"
    stray.mkdir()
    write_attempt(stray, losses(1.0), writer="tb")

    discovery = discover_runs(tmp_path)

    assert len(discovery.runs) == 1
    (problem,) = discovery.problems
    assert problem.startswith("other/tb/events.out.tfevents.")
    assert "in no session_* directory" in problem


def test_a_session_without_event_files_is_listed_with_a_problem(tmp_path):
    write_run(tmp_path / "with_events", losses(1.0), config())
    quiet = tmp_path / "no_events" / SESSION
    quiet.mkdir(parents=True)
    (quiet / "config.yaml").write_text("session_config: {rng_seed: 1}\n")

    summary = _summary(tmp_path)

    assert f"no_events/{SESSION}: no TensorBoard event files" in summary.problems
    assert f"no_events/{SESSION}: loss is not written by this run" in summary.problems


def test_no_runs_found_is_an_error(tmp_path):
    with pytest.raises(ReportError, match="no runs found under"):
        _summary(tmp_path)


# -- reading scalars ---------------------------------------------------------------


def test_every_point_is_read_past_tensorboards_sampling_limit(tmp_path):
    # EventAccumulator keeps a 10 000-point sample per tag by default.
    count = 12_345
    points = [(step, float(step % 1000)) for step in range(1, count + 1)]
    points[7_000] = (7_001, -5.0)
    write_run(tmp_path, {"loss": points}, config())

    (run,) = _summary(tmp_path).runs
    loss = run.metrics["loss"]

    assert loss.points == count
    assert (loss.last, loss.last_step) == (float(count % 1000), count)
    assert (loss.best, loss.best_step) == (-5.0, 7_001)


def test_a_resume_rewrites_steps_and_the_later_attempt_wins(tmp_path):
    session_dir = write_run(tmp_path, losses(8.0, 4.0, 2.0, 1.0), config(), wall_time=100.0)
    # Resumed from the checkpoint at step 2: steps 3 and 4 are written again.
    write_attempt(session_dir, losses(3.0, 0.5, 0.25, start=3), wall_time=500.0)

    (run,) = _summary(tmp_path).runs

    assert [(p.step, p.value) for p in run.run.scalars["loss"]] == [
        (1, 8.0), (2, 4.0), (3, 3.0), (4, 0.5), (5, 0.25),
    ]
    assert run.run.rewritten == {"loss": 2}
    assert len(run.run.attempts) == 2
    assert (run.metrics["loss"].best, run.metrics["loss"].rewritten) == (0.25, 2)


def test_attempts_are_ordered_by_when_they_ran_not_by_file_name(tmp_path):
    session_dir = write_run(tmp_path, losses(1.0, 2.0), config(), wall_time=900.0)
    # Written second, but its points are older: it is the first attempt.
    write_attempt(session_dir, losses(7.0, 7.0), wall_time=100.0)

    (run,) = _summary(tmp_path).runs

    assert [p.value for p in run.run.scalars["loss"]] == [1.0, 2.0]


def test_two_writers_prefix_tags_with_their_directory(tmp_path):
    session_dir = write_run(tmp_path, losses(1.0), config(), writer="Tensorboard_tensorboard_a")
    write_attempt(session_dir, losses(2.0), writer="Tensorboard_tensorboard_b")

    (run,) = discover_runs(tmp_path).runs

    assert sorted(run.scalars) == [
        "Tensorboard_tensorboard_a/loss", "Tensorboard_tensorboard_b/loss",
    ]


def test_a_truncated_event_file_keeps_what_came_before_and_says_so(tmp_path):
    session_dir = write_run(tmp_path, losses(4.0, 2.0, 1.0), config())
    (event_file,) = (session_dir / WRITER).glob("events.out.tfevents.*")
    event_file.write_bytes(event_file.read_bytes()[:-5])

    summary = _summary(tmp_path)

    assert [p.value for p in summary.runs[0].run.scalars["loss"]] == [4.0, 2.0]
    (problem,) = summary.problems
    assert "truncated record" in problem and event_file.name in problem


def test_a_corrupt_record_stops_the_read_and_says_where(tmp_path):
    session_dir = write_run(tmp_path, losses(4.0, 2.0, 1.0), config())
    (event_file,) = (session_dir / WRITER).glob("events.out.tfevents.*")
    data = bytearray(event_file.read_bytes())
    data[-10] ^= 0xFF
    event_file.write_bytes(bytes(data))

    summary = _summary(tmp_path)

    assert [p.value for p in summary.runs[0].run.scalars["loss"]] == [4.0, 2.0]
    (problem,) = summary.problems
    assert "corrupt record (CRC mismatch) at byte" in problem


def test_tensor_valued_scalars_are_read(tmp_path):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    session_dir = tmp_path / SESSION
    session_dir.mkdir()
    (session_dir / "config.yaml").write_text("session_config: {rng_seed: 1}\n")
    writer = SummaryWriter(log_dir=str(session_dir / WRITER))
    for step, value in ((1, 2.0), (2, 0.5)):
        writer.add_scalar("loss", torch.tensor(value), step, new_style=True)
    writer.close()

    (run,) = discover_runs(tmp_path).runs

    assert [(p.step, p.value) for p in run.scalars["loss"]] == [(1, 2.0), (2, 0.5)]


# -- hyperparameters -------------------------------------------------------------


def test_only_differing_settings_are_columns_and_the_rest_are_shared(tmp_path):
    for lr in (0.5, 0.25, 0.125):
        write_run(tmp_path / f"lr{lr}", losses(1.0), config(lr=lr, sessions_dir=str(tmp_path / f"lr{lr}")))

    summary = _summary(tmp_path)

    assert [column.name for column in summary.columns] == ["optimizer.optimizer.kwargs.lr"]
    assert {run.run_id: run.hyperparameters["optimizer.optimizer.kwargs.lr"] for run in summary.runs} == {
        f"lr0.5/{SESSION}": 0.5, f"lr0.25/{SESSION}": 0.25, f"lr0.125/{SESSION}": 0.125,
    }
    assert summary.shared["optimizer.optimizer.kind"] == "sgd"
    assert summary.ignored_keys == list(DEFAULT_IGNORED_KEYS)
    assert not any("sessions_dir" in path for path in summary.shared)


def test_the_default_ignores_can_be_cleared_and_keys_added(tmp_path):
    for lr in (0.5, 0.25):
        write_run(tmp_path / f"lr{lr}", losses(1.0), config(lr=lr, sessions_dir=str(tmp_path / f"lr{lr}")))
    discovery = discover_runs(tmp_path)

    everything = summarize_runs(discovery, ["loss"], ignored_keys=())
    without_optimizer = summarize_runs(
        discovery, ["loss"], ignored_keys=[*DEFAULT_IGNORED_KEYS, "optimizer"],
    )

    assert [c.name for c in everything.columns] == [
        "optimizer.optimizer.kwargs.lr", "session_config.sessions_dir",
    ]
    assert without_optimizer.columns == []


@pytest.mark.parametrize("listed", ["top level", "resources group"])
def test_the_launch_rendezvous_is_not_compared_but_world_size_is(tmp_path, listed):
    # The engine picks a free port at every launch and writes it to config.yaml.
    for port, world_size in ((40001, 1), (40002, 2)):
        ddp = {"backend": "gloo", "master_addr": "127.0.0.1", "master_port": str(port), "world_size": world_size}
        extra = {"ddp": ddp} if listed == "top level" else {"resources": {"ddp": ddp}}
        write_run(tmp_path / f"p{port}", losses(1.0), config(**extra))

    summary = _summary(tmp_path)

    prefix = "ddp" if listed == "top level" else "resources.ddp"
    assert [c.name for c in summary.columns] == [f"{prefix}.world_size"]


def test_a_setting_missing_in_some_runs_is_a_column(tmp_path):
    write_run(tmp_path / "a", losses(1.0), config(clip={"max_norm": 1.0}))
    write_run(tmp_path / "b", losses(1.0), config())

    summary = _summary(tmp_path, group_by_seed=False)

    assert {run.run_id: run.hyperparameters["clip.max_norm"] for run in summary.runs} == {
        f"a/{SESSION}": 1.0, f"b/{SESSION}": "absent",
    }


def test_values_of_different_types_are_different(tmp_path):
    write_run(tmp_path / "a", losses(1.0), config(flag=1))
    write_run(tmp_path / "b", losses(1.0), config(flag=True))

    summary = _summary(tmp_path, group_by_seed=False)

    assert [c.name for c in summary.columns] == ["flag"]


def test_keys_that_are_not_plain_names_are_quoted_and_can_be_ignored(tmp_path):
    write_run(tmp_path / "a", losses(1.0), config(**{"projection_head#student": {"out": 8}}))
    write_run(tmp_path / "b", losses(1.0), config(**{"projection_head#student": {"out": 16}}))
    discovery = discover_runs(tmp_path)

    summary = summarize_runs(discovery, ["loss"], group_by_seed=False)
    ignored = summarize_runs(discovery, ["loss"], group_by_seed=False,
                             ignored_keys=['["projection_head#student"]'])

    assert [c.name for c in summary.columns] == ['["projection_head#student"].out']
    assert ignored.columns == []


@pytest.mark.parametrize("key", ["a..b", "a[0]b", ".a", "a.", 'a["b'])
def test_a_malformed_ignore_key_is_refused(tmp_path, key):
    write_run(tmp_path, losses(1.0), config())

    with pytest.raises(ReportError, match="cannot read config path"):
        _summary(tmp_path, ignored_keys=[key])


def test_a_run_without_a_config_has_unknown_hyperparameters(tmp_path):
    write_run(tmp_path / "a", losses(1.0), config(lr=0.5))
    write_run(tmp_path / "b", losses(1.0), config(lr=0.25))
    write_run(tmp_path / "c", losses(1.0), None)

    summary = _summary(tmp_path)
    runs = _by_id(summary)

    assert runs[f"c/{SESSION}"].hyperparameters["optimizer.optimizer.kwargs.lr"] == "unknown"
    assert f"c/{SESSION}: no config.yaml; its hyperparameters are unknown" in summary.problems
    # Unknown hyperparameters match no other run.
    assert len({run.group for run in summary.runs}) == 3


def test_runs_without_a_config_are_not_grouped_together(tmp_path):
    write_run(tmp_path / "a", losses(1.0), None)
    write_run(tmp_path / "b", losses(1.0), None)
    write_run(tmp_path / "c", losses(1.0), config())

    summary = _summary(tmp_path)

    assert [len(group.members) for group in summary.groups] == [1, 1, 1]


def test_an_extended_run_is_flagged(tmp_path):
    session_dir = write_run(tmp_path, losses(1.0), config())
    (session_dir / "config_extension_20261031_130000.yaml").write_text("{}\n")

    (run,) = _summary(tmp_path).runs

    assert run.run.extensions == 1
    assert "| 1 |" in render_markdown(_summary(tmp_path))


def test_an_hpc_task_is_named_by_its_task_and_shows_its_matrix_values(tmp_path):
    batch = tmp_path / "vit-probes-20261014"
    for index, (lr, seed) in enumerate(((0.5, 41), (0.5, 42), (0.25, 41))):
        task_dir = batch / "tasks" / "probes" / "frozen" / f"{index:03d}"
        write_run(task_dir / "sessions", losses(2.0, 1.0 - lr / 4 + seed / 1024),
                  config(lr=lr, seed=seed, sessions_dir=str(task_dir / "sessions")))
        (task_dir / "variant.json").write_text(json.dumps([
            {"address": {"component": "optimizer", "path": ["optimizer", "kwargs", "lr"]}, "value": lr},
            {"address": {"path": ["session_config", "rng_seed"]}, "value": seed},
        ]))

    summary = _summary(tmp_path)

    assert sorted(_by_id(summary)) == ["probes/frozen/000", "probes/frozen/001", "probes/frozen/002"]
    matrix = [c for c in summary.columns if c.source == "matrix"]
    assert [(c.name, c.seed) for c in matrix] == [
        ("optimizer:optimizer.kwargs.lr", False), ("session_config.rng_seed", True),
    ]
    groups = {tuple(g.members): g for g in summary.groups}
    assert set(groups) == {("probes/frozen/000", "probes/frozen/001"), ("probes/frozen/002",)}
    assert groups[("probes/frozen/000", "probes/frozen/001")].seeds == ["41", "42"]


def test_a_task_with_two_session_directories_reports_both(tmp_path):
    task_dir = tmp_path / "tasks" / "g" / "e" / "000"
    write_run(task_dir / "sessions", losses(1.0), config(), session="session_20261031_120000")
    write_run(task_dir / "sessions", losses(1.0), config(), session="session_20261031_130000")
    (task_dir / "variant.json").write_text("[]")

    summary = _summary(tmp_path)

    assert sorted(_by_id(summary)) == [
        "g/e/000/session_20261031_120000", "g/e/000/session_20261031_130000",
    ]
    assert any("task g/e/000 has 2 session directories" in p for p in summary.problems)


# -- metrics ---------------------------------------------------------------------


def test_best_needs_a_direction(tmp_path):
    write_run(tmp_path, losses(4.0, 1.0, 2.0), config())

    (bare,) = _summary(tmp_path, metrics=["loss"]).runs
    (lowest,) = _summary(tmp_path, metrics=["loss:min"]).runs
    (highest,) = _summary(tmp_path, metrics=[Metric("loss", "max")]).runs

    assert (bare.metrics["loss"].best, bare.metrics["loss"].last) == (None, 2.0)
    assert (lowest.metrics["loss"].best, lowest.metrics["loss"].best_step) == (1.0, 2)
    assert (highest.metrics["loss"].best, highest.metrics["loss"].best_step) == (4.0, 1)
    with pytest.raises(ReportError, match="direction must be min or max"):
        Metric("loss", "lowest")


def test_sorting_needs_a_metric_with_a_direction(tmp_path):
    write_run(tmp_path, losses(1.0), config())

    with pytest.raises(ReportError, match="give it a direction"):
        _summary(tmp_path, metrics=["loss"], sort="loss")
    with pytest.raises(ReportError, match="not a --metric"):
        _summary(tmp_path, sort="accuracy")


def test_a_metric_no_run_writes_is_an_error_listing_the_tags(tmp_path):
    write_run(tmp_path, {"loss": [(1, 1.0)], "acc": [(1, 0.5)]}, config())

    with pytest.raises(ReportError, match="'accuracy'; the tags written are: acc, loss"):
        _summary(tmp_path, metrics=["accuracy:max"])
    with pytest.raises(ReportError, match="no tag matches 'val/\\*'"):
        _summary(tmp_path, metrics=(), tags=["val/*"])


def test_a_metric_given_twice_with_different_directions_is_refused(tmp_path):
    write_run(tmp_path, losses(1.0), config())

    with pytest.raises(ReportError, match="given twice"):
        _summary(tmp_path, metrics=["loss:min", "loss:max"])


def test_without_metrics_every_tag_is_reported_last_only(tmp_path):
    write_run(tmp_path, {"loss": [(1, 1.0)], "optimizer/lr/group_0": [(1, 0.5)]}, config())

    summary = _summary(tmp_path, metrics=())

    assert [(m.tag, m.direction) for m in summary.metrics] == [
        ("loss", None), ("optimizer/lr/group_0", None),
    ]


def test_tag_globs_add_tags(tmp_path):
    write_run(tmp_path, {"loss": [(1, 1.0)], "dino/center_norm": [(1, 0.5)], "dino/teacher_temp": [(1, 0.25)]}, config())

    summary = _summary(tmp_path, metrics=["loss:min"], tags=["dino/*"])

    assert [m.tag for m in summary.metrics] == ["loss", "dino/center_norm", "dino/teacher_temp"]


def test_mean_of_the_last_n_points(tmp_path):
    write_run(tmp_path, losses(8.0, 4.0, 2.0, 1.0), config())

    (run,) = _summary(tmp_path, last_n=3).runs

    assert run.metrics["loss"].mean_last_n == pytest.approx((4.0 + 2.0 + 1.0) / 3)
    with pytest.raises(ReportError, match="last_n must be a positive integer"):
        _summary(tmp_path, last_n=0)


def test_a_tag_missing_in_one_run_is_a_problem(tmp_path):
    write_run(tmp_path / "a", {"loss": [(1, 1.0)], "acc": [(1, 0.5)]}, config(lr=0.5))
    write_run(tmp_path / "b", {"loss": [(1, 1.0)]}, config(lr=0.25))

    summary = _summary(tmp_path, metrics=["acc:max"])

    assert "acc" not in _by_id(summary)[f"b/{SESSION}"].metrics
    assert f"b/{SESSION}: acc is not written by this run" in summary.problems
    assert "missing" in render_markdown(summary)


def test_non_finite_values_are_shown_left_out_of_best_and_reported(tmp_path):
    write_run(tmp_path, losses(2.0, 1.0, math.nan, math.inf), config())

    summary = _summary(tmp_path)
    (run,) = summary.runs

    assert math.isinf(run.metrics["loss"].last)
    assert (run.metrics["loss"].best, run.metrics["loss"].nonfinite) == (1.0, 2)
    assert f"{SESSION}: loss: 2 non-finite value(s) (nan/inf)" in summary.problems


# -- seed groups -----------------------------------------------------------------


def test_runs_that_differ_only_in_seed_form_a_group_with_mean_and_std(tmp_path):
    _lr_runs(tmp_path, seeds=(1, 2, 3))

    summary = _summary(tmp_path, sort="loss")

    assert [g.name for g in summary.groups] == ["G1", "G2"]
    by_lr = {g.hyperparameters["optimizer.optimizer.kwargs.lr"]: g for g in summary.groups}
    assert set(by_lr) == {0.5, 0.25}
    for lr, group in by_lr.items():
        lasts = [1.0 + lr + seed / 8 for seed in (1, 2, 3)]
        stats = group.stats[("loss", "last")]
        assert stats.mean == pytest.approx(statistics.fmean(lasts))
        assert stats.std == pytest.approx(statistics.stdev(lasts))
        assert (stats.min, stats.max, stats.n) == (min(lasts), max(lasts), 3)
        assert group.seeds == ["1", "2", "3"]
        assert "optimizer.optimizer.kwargs.lr" in group.hyperparameters
        assert "session_config.rng_seed" not in group.hyperparameters
    # Sorted by the group mean of the best loss: lr 0.25 is lower.
    assert summary.groups[0].hyperparameters["optimizer.optimizer.kwargs.lr"] == 0.25
    assert {run.group for run in summary.runs if run.run_id.startswith("lr0.25")} == {"G1"}
    assert summary.problems == []


def test_a_group_of_one_has_no_std(tmp_path):
    _lr_runs(tmp_path, seeds=(1,))

    summary = _summary(tmp_path)

    for group in summary.groups:
        stats = group.stats[("loss", "last")]
        assert (stats.n, stats.std) == (1, None)
    assert summary.problems == []


def test_members_that_end_at_different_steps_are_a_problem(tmp_path):
    write_run(tmp_path / "s1", losses(2.0, 1.0), config(seed=1))
    write_run(tmp_path / "s2", losses(2.0, 1.0, 0.5), config(seed=2))

    summary = _summary(tmp_path)

    (problem,) = summary.problems
    assert problem.startswith("group G1: loss ends at different steps")
    assert f"s1/{SESSION}: 2" in problem and f"s2/{SESSION}: 3" in problem


def test_two_members_with_the_same_seed_are_a_problem(tmp_path):
    write_run(tmp_path / "first", losses(1.0), config(seed=7))
    write_run(tmp_path / "again", losses(1.0), config(seed=7))

    summary = _summary(tmp_path)

    (problem,) = summary.problems
    assert problem.startswith("group G1: runs ") and "have the same seed (7)" in problem


def test_non_finite_member_values_are_left_out_of_group_stats_and_counted(tmp_path):
    write_run(tmp_path / "s1", losses(2.0, 1.0), config(seed=1))
    write_run(tmp_path / "s2", losses(2.0, 0.5), config(seed=2))
    write_run(tmp_path / "s3", losses(2.0, math.nan), config(seed=3))

    (group,) = _summary(tmp_path).groups
    stats = group.stats[("loss", "last")]

    assert (stats.n, stats.nonfinite, stats.mean) == (2, 1, 0.75)


def test_an_unknown_seed_key_is_an_error(tmp_path):
    _lr_runs(tmp_path)

    with pytest.raises(ReportError, match="seed key\\(s\\) data_manager.seed are in no run"):
        _summary(tmp_path, seed_keys=["data_manager.seed"])


def test_a_seed_key_equal_in_every_run_is_still_known(tmp_path):
    _lr_runs(tmp_path, seeds=(5,))

    summary = _summary(tmp_path)

    assert summary.shared["session_config.rng_seed"] == 5
    assert all(group.seeds == ["5"] for group in summary.groups)


def test_several_seed_keys(tmp_path):
    for model_seed, data_seed in ((1, 10), (2, 20)):
        write_run(tmp_path / f"m{model_seed}", losses(1.0),
                  config(seed=model_seed, data_manager={"seed": data_seed}))

    summary = _summary(tmp_path, seed_keys=["session_config.rng_seed", "data_manager.seed"])

    (group,) = summary.groups
    assert group.seeds == [
        "data_manager.seed=10, session_config.rng_seed=1",
        "data_manager.seed=20, session_config.rng_seed=2",
    ]


def test_grouping_can_be_turned_off(tmp_path):
    _lr_runs(tmp_path, seeds=(1, 2))

    summary = _summary(tmp_path, group_by_seed=False)

    assert summary.groups is None
    assert all(run.group is None for run in summary.runs)
    assert "## Seed groups" not in render_markdown(summary)


# -- rendering -------------------------------------------------------------------


def test_markdown_has_groups_runs_shared_settings_and_problems(tmp_path):
    _lr_runs(tmp_path, seeds=(1, 2))

    text = render_markdown(_summary(tmp_path, last_n=2))

    for heading in ("# Run summary", "## Seed groups", "## Runs", "## Shared settings", "## Problems"):
        assert heading in text
    assert "| group | optimizer.optimizer.kwargs.lr | runs | seeds | loss last | loss best | loss mean2 |" in text
    assert "session_config.max_iterations = 10" in text
    assert "Not compared (ignored keys): `session_config.sessions_dir`, `ddp.master_addr`" in text


def test_markdown_escapes_table_cells(tmp_path):
    write_run(tmp_path / "a", losses(1.0), config(note="a|b"))
    write_run(tmp_path / "b", losses(1.0), config(note="c"))

    text = render_markdown(_summary(tmp_path, group_by_seed=False))

    assert "a\\|b" in text


def test_csv_tables(tmp_path):
    _lr_runs(tmp_path, seeds=(1, 2))
    summary = _summary(tmp_path)

    groups = list(csv.DictReader(io.StringIO(render_csv(summary))))
    runs = list(csv.DictReader(io.StringIO(render_csv(summary, table="runs"))))

    assert len(groups) == 2 and len(runs) == 4
    assert float(groups[0]["loss last mean"]) == pytest.approx(summary.groups[0].stats[("loss", "last")].mean)
    assert {row["loss best step"] for row in runs} == {"3"}
    with pytest.raises(ValueError, match="no groups table"):
        render_csv(_summary(tmp_path, group_by_seed=False))


def test_json_is_valid_even_with_non_finite_values(tmp_path):
    write_run(tmp_path, losses(1.0, math.nan), config())

    document = json.loads(render_json(_summary(tmp_path)))

    assert document["runs"][0]["metrics"]["loss"]["last"] == "nan"
    assert document["groups"][0]["members"] == [SESSION]


def test_a_step_written_twice_in_one_attempt_is_a_problem(tmp_path):
    write_run(tmp_path, {"loss": [(1, 2.0), (2, 1.0), (2, 0.5)]}, config())

    summary = _summary(tmp_path)

    assert [p.value for p in summary.runs[0].run.scalars["loss"]] == [2.0, 0.5]
    assert summary.runs[0].run.rewritten == {}
    (problem,) = summary.problems
    assert "loss is written twice at step 2 in one attempt" in problem


def test_an_unreadable_variant_is_a_problem(tmp_path):
    task_dir = tmp_path / "tasks" / "g" / "e" / "000"
    write_run(task_dir / "sessions", losses(1.0), config())
    (task_dir / "variant.json").write_text("{not json")

    summary = _summary(tmp_path)

    assert summary.runs[0].run_id == "g/e/000"
    assert any("variant.json cannot be read" in p for p in summary.problems)


def _component_seed_tasks(root, *, grouped):
    """Two tasks that differ only in a seed the matrix sets on the
    `data_manager` component, listed at the top level or in `resources`."""
    for index, seed in enumerate((10, 20)):
        task_dir = root / "tasks" / "g" / "e" / f"{index:03d}"
        data_manager = {"batch_size": 4, "seed": seed}
        extra = {"resources": {"data_manager": data_manager}} if grouped else {"data_manager": data_manager}
        write_run(task_dir / "sessions", losses(1.0), config(**extra))
        (task_dir / "variant.json").write_text(json.dumps([
            {"address": {"component": "data_manager", "path": ["seed"]}, "value": seed},
        ]))


@pytest.mark.parametrize("grouped", [False, True], ids=["top level", "resources group"])
@pytest.mark.parametrize("seed_key", ["config path", "matrix label"])
def test_a_matrix_seed_on_a_component_groups_its_runs(tmp_path, grouped, seed_key):
    _component_seed_tasks(tmp_path, grouped=grouped)
    if seed_key == "matrix label":
        key = "data_manager:seed"
    else:
        key = "resources.data_manager.seed" if grouped else "data_manager.seed"

    summary = _summary(tmp_path, seed_keys=[key])

    (group,) = summary.groups
    assert group.members == ["g/e/000", "g/e/001"]
    assert len(group.seeds) == 2 and group.seeds[0] != group.seeds[1]
    assert [(c.name, c.seed) for c in summary.columns if c.source == "matrix"] == [
        ("data_manager:seed", True),
    ]
    assert summary.problems == []


def test_a_matrix_component_missing_from_the_config_is_a_problem(tmp_path):
    task_dir = tmp_path / "tasks" / "g" / "e" / "000"
    write_run(task_dir / "sessions", losses(1.0), config())
    (task_dir / "variant.json").write_text(json.dumps([
        {"address": {"component": "data_manager", "path": ["seed"]}, "value": 1},
    ]))

    summary = _summary(tmp_path)

    assert any(
        "matrix value data_manager:seed: component 'data_manager' is not in config.yaml" in p
        for p in summary.problems
    )


def test_shared_settings_say_which_runs_have_no_config(tmp_path):
    write_run(tmp_path / "known", losses(1.0), config(lr=0.5))
    write_run(tmp_path / "lost", losses(1.0), None)

    summary = _summary(tmp_path)
    text = render_markdown(summary)

    assert summary.unknown_config_runs == [f"lost/{SESSION}"]
    assert summary.shared["optimizer.optimizer.kwargs.lr"] == 0.5
    assert f"Equal in the 1 run(s) with a config; unknown for `lost/{SESSION}`" in text
    assert json.loads(render_json(summary))["unknown_config_runs"] == [f"lost/{SESSION}"]


def test_with_every_config_known_shared_settings_have_no_caveat(tmp_path):
    write_run(tmp_path / "a", losses(1.0), config(lr=0.5))
    write_run(tmp_path / "b", losses(1.0), config(lr=0.25))

    summary = _summary(tmp_path)

    assert summary.unknown_config_runs == []
    assert "unknown for" not in render_markdown(summary)


def test_a_component_seed_key_resolves_in_local_runs_too(tmp_path):
    # No matrix: the key names the component's entry wherever it is listed.
    for seed in (10, 20):
        write_run(tmp_path / f"s{seed}", losses(1.0),
                  config(resources={"data_manager": {"batch_size": 4, "seed": seed}}))

    summary = _summary(tmp_path, seed_keys=["data_manager:seed"])

    (group,) = summary.groups
    assert group.seeds == ["10", "20"]
    assert [(c.name, c.seed) for c in summary.columns] == [("resources.data_manager.seed", True)]


def test_a_malformed_component_seed_key_is_refused(tmp_path):
    write_run(tmp_path, losses(1.0), config())

    with pytest.raises(ReportError, match="seed key: cannot read config path 'a..b'"):
        _summary(tmp_path, seed_keys=["data_manager:a..b"])
