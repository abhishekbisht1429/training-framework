"""A `Summary` as Markdown, CSV, or JSON."""

from __future__ import annotations

import csv
import io
import json
import math
import posixpath
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from training_framework.reporting.summary import (
    GroupSummary,
    RunSummary,
    Stats,
    Summary,
)

TABLES = ("groups", "runs")


def render_markdown(summary: Summary, *, generated: datetime | None = None) -> str:
    generated = generated or datetime.now(timezone.utc)
    lines = ["# Run summary", ""]
    lines.append("- Roots: " + ", ".join(f"`{root}`" for root in summary.roots))
    lines.append(f"- Generated: {generated.strftime('%Y-%m-%d %H:%M %Z').strip()}")
    count = f"- Runs: {len(summary.runs)}"
    if summary.groups is not None:
        count += (
            f"; seed groups: {len(summary.groups)} (seed keys: "
            + ", ".join(f"`{key}`" for key in summary.seed_keys) + ")"
        )
    else:
        count += "; not grouped by seed"
    lines.append(count)
    lines.append("- Metrics: " + ", ".join(
        f"`{metric.tag}` ({metric.direction or 'last only'})" for metric in summary.metrics
    ))
    if summary.sort is not None:
        lines.append(f"- Sorted by: best `{summary.sort.tag}` ({summary.sort.direction})")
    lines.append(
        "- `last` is the value at the run's last logged step, `best` the "
        "lowest/highest finite value"
        + (f", `mean{summary.last_n}` the mean of the last {summary.last_n} points" if summary.last_n else "")
        + "; `value @ step`."
    )
    lines.append(
        "- Hyperparameters are from each run's `config.yaml` (its latest "
        "settings: an extended run's changed partway, see `extended`) and, "
        "for HPC tasks, `variant.json` (`component:path` columns)."
    )
    lines.append("")

    if summary.groups is not None:
        lines += ["## Seed groups", ""]
        lines += _markdown_table(*_group_rows(summary))
        lines.append("")
    lines += ["## Runs", ""]
    lines += _markdown_table(*_run_rows(summary))
    lines.append("")

    lines += ["## Shared settings", ""]
    if summary.unknown_config_runs:
        known = len(summary.runs) - len(summary.unknown_config_runs)
        lines += [
            f"Equal in the {known} run(s) with a config; unknown for "
            + ", ".join(f"`{run_id}`" for run_id in summary.unknown_config_runs)
            + " (no readable `config.yaml`).",
            "",
        ]
    if summary.shared:
        lines.append("```text")
        lines += [f"{path} = {_text(value)}" for path, value in summary.shared.items()]
        lines.append("```")
    else:
        lines.append("None: every setting differs in some run.")
    if summary.ignored_keys:
        lines += ["", "Not compared (ignored keys): " + ", ".join(f"`{key}`" for key in summary.ignored_keys)]
    lines += ["", "## Problems", ""]
    if summary.problems:
        lines += [f"- {_escape_inline(problem)}" for problem in summary.problems]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def render_csv(summary: Summary, *, table: str = "groups") -> str:
    if table not in TABLES:
        raise ValueError(f"table must be one of {', '.join(TABLES)}; got {table!r}")
    if table == "groups":
        if summary.groups is None:
            raise ValueError("the runs are not grouped by seed; there is no groups table")
        header, rows = _group_rows(summary, plain=True)
    else:
        header, rows = _run_rows(summary, plain=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def render_json(summary: Summary) -> str:
    def run_json(run: RunSummary) -> dict[str, Any]:
        record = run.run
        return {
            "run": run.run_id,
            "group": run.group,
            "task": record.task,
            "session_dir": record.session_dir,
            "hyperparameters": run.hyperparameters,
            "extensions": record.extensions,
            "writers": record.writers,
            "attempts": [asdict(attempt) for attempt in record.attempts],
            "metrics": {tag: asdict(metric) for tag, metric in run.metrics.items()},
        }

    def group_json(group: GroupSummary) -> dict[str, Any]:
        stats: dict[str, dict[str, Any]] = {}
        for (tag, field), value in group.stats.items():
            stats.setdefault(tag, {})[field] = asdict(value)
        return {
            "group": group.name,
            "hyperparameters": group.hyperparameters,
            "members": group.members,
            "seeds": group.seeds,
            "stats": stats,
        }

    document = {
        "roots": summary.roots,
        "metrics": [asdict(metric) for metric in summary.metrics],
        "last_n": summary.last_n,
        "sort": None if summary.sort is None else asdict(summary.sort),
        "seed_keys": summary.seed_keys,
        "columns": [asdict(column) for column in summary.columns],
        "groups": None if summary.groups is None else [group_json(g) for g in summary.groups],
        "runs": [run_json(run) for run in summary.runs],
        "shared": summary.shared,
        "unknown_config_runs": summary.unknown_config_runs,
        "ignored_keys": summary.ignored_keys,
        "problems": summary.problems,
    }
    return json.dumps(_json_safe(document), indent=2) + "\n"


# -- tables ----------------------------------------------------------------------


def _field_label(summary: Summary, field: str) -> str:
    return f"mean{summary.last_n}" if field == "mean_last_n" else field


def _metric_fields(summary: Summary, metric) -> list[str]:
    fields = ["last"]
    if metric.direction is not None:
        fields.append("best")
    if summary.last_n is not None:
        fields.append("mean_last_n")
    return fields


def _run_rows(summary: Summary, plain: bool = False):
    header = ["run"]
    if summary.groups is not None:
        header.append("group")
    header += [column.name for column in summary.columns]
    header += ["extended", "attempts"]
    for metric in summary.metrics:
        for field in _metric_fields(summary, metric):
            label = f"{metric.tag} {_field_label(summary, field)}"
            if plain:
                header.append(label)
                if field in ("last", "best"):
                    header.append(f"{label} step")
            else:
                header.append(label)
        if plain:
            header += [f"{metric.tag} points", f"{metric.tag} rewritten"]
    rows = []
    for run in summary.runs:
        row: list[Any] = [run.run_id]
        if summary.groups is not None:
            row.append(run.group)
        row += [_cell(run.hyperparameters[column.name], plain) for column in summary.columns]
        row.append(run.run.extensions if plain else (run.run.extensions or "-"))
        rewritten = max(run.run.rewritten.values(), default=0)
        if plain or not rewritten:
            row.append(_attempts(run))
        else:
            row.append(f"{_attempts(run)} ({rewritten} steps rewritten)")
        for metric in summary.metrics:
            values = run.metrics.get(metric.tag)
            for field in _metric_fields(summary, metric):
                if values is None:
                    row += ["", ""] if plain and field in ("last", "best") else ["" if plain else "missing"]
                    continue
                value = getattr(values, field)
                if field in ("last", "best"):
                    step = getattr(values, f"{field}_step")
                    if plain:
                        row += [_number(value, plain), "" if step is None else step]
                    else:
                        row.append("-" if value is None else f"{_number(value)} @ {step}")
                else:
                    row.append(_number(value, plain))
            if plain:
                row += ["", ""] if values is None else [values.points, values.rewritten]
        rows.append(row)
    return header, rows


def _group_rows(summary: Summary, plain: bool = False):
    plain_columns = [column for column in summary.columns if not column.seed]
    header = ["group"] + [column.name for column in plain_columns] + ["runs", "seeds"]
    names = []
    for metric in summary.metrics:
        for field in _metric_fields(summary, metric):
            label = f"{metric.tag} {_field_label(summary, field)}"
            names.append((metric.tag, field))
            if plain:
                header += [f"{label} {part}" for part in ("mean", "std", "min", "max", "n", "nonfinite", "missing")]
            else:
                header.append(label)
    if plain:
        header.append("members")
    rows = []
    for group in summary.groups or []:
        row: list[Any] = [group.name]
        row += [_cell(group.hyperparameters[column.name], plain) for column in plain_columns]
        row += [len(group.members), ("; " if plain else ", ").join(group.seeds)]
        for key in names:
            stats = group.stats[key]
            if plain:
                row += [_number(stats.mean, True), _number(stats.std, True), _number(stats.min, True),
                        _number(stats.max, True), stats.n, stats.nonfinite, stats.missing]
            else:
                row.append(_stats_text(stats))
        if plain:
            row.append("; ".join(group.members))
        rows.append(row)
    return header, rows


def _attempts(run: RunSummary) -> int:
    """Launches of the run: event files per writer directory (one writer can
    be added by a resume, so the most any writer has)."""
    per_writer: dict[str, int] = {}
    for attempt in run.run.attempts:
        writer = posixpath.dirname(attempt.path)
        per_writer[writer] = per_writer.get(writer, 0) + 1
    return max(per_writer.values(), default=0)


def _stats_text(stats: Stats) -> str:
    if stats.mean is None:
        text = "-"
    elif stats.std is None:
        text = f"{_number(stats.mean)}"
    else:
        # Three significant digits: the spread, not float32 rounding noise.
        text = f"{_number(stats.mean)} ± {stats.std:.3g}"
    text += f" (n={stats.n})"
    if stats.nonfinite:
        text += f", {stats.nonfinite} non-finite left out"
    if stats.missing:
        text += f", {stats.missing} missing"
    return text


def _markdown_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def line(cells):
        return "| " + " | ".join(_escape_cell(str(cell)) for cell in cells) + " |"
    return [line(header), "|" + "|".join("---" for _ in header) + "|"] + [line(row) for row in rows]


def _escape_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _escape_inline(text: str) -> str:
    return text.replace("\n", " ")


def _number(value: float | None, plain: bool = False) -> Any:
    if value is None:
        return "" if plain else "-"
    if plain:
        return repr(value) if math.isfinite(value) else str(value)
    return f"{value:.6g}"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=repr)


def _cell(value: Any, plain: bool) -> Any:
    if isinstance(value, float) and not plain:
        return _number(value)
    return _text(value)


def _json_safe(value: Any) -> Any:
    """NaN and infinity as strings, so the output is valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
