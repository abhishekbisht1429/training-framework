"""`python -m training_framework.reporting summarize ROOT [ROOT ...]`."""

from __future__ import annotations

import argparse
import os
import sys

from training_framework.reporting.render import TABLES, render_csv, render_json, render_markdown
from training_framework.reporting.runs import ReportError, discover_runs
from training_framework.reporting.summary import (
    DEFAULT_IGNORED_KEYS,
    DEFAULT_SEED_KEYS,
    summarize_runs,
)

#: Exit status when the report was written but has problems and --strict
#: was given.
EXIT_PROBLEMS = 1
#: Exit status when no report could be made.
EXIT_ERROR = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m training_framework.reporting")
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser(
        "summarize",
        help="compare runs: their differing hyperparameters and TensorBoard scalars",
        description=(
            "Find every run (session_* directory) under the given directories -- "
            "local sessions_dirs, HPC batch directories, or `hpc fetch` results -- "
            "and write one report comparing them: a row per run with the "
            "hyperparameters that differ and each metric, and runs that differ "
            "only in their seed grouped into mean ± std."
        ),
    )
    summarize.add_argument("roots", nargs="+", metavar="ROOT",
                           help="a directory to look for runs in")
    summarize.add_argument("--metric", action="append", default=[], metavar="TAG[:min|:max]",
                           help="a scalar tag to report; with :min or :max its best value too "
                                "(repeatable; default: every tag, last value only)")
    summarize.add_argument("--tags", action="append", default=[], metavar="GLOB",
                           help="also report every tag matching GLOB, last value only (repeatable)")
    summarize.add_argument("--last-n", type=int, metavar="N",
                           help="also report the mean of each tag's last N points")
    summarize.add_argument("--sort", metavar="TAG",
                           help="order runs and groups by the best value of this --metric")
    summarize.add_argument("--include", action="append", default=[], metavar="GLOB",
                           help="only runs whose id matches GLOB (repeatable)")
    summarize.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                           help="leave out runs whose id matches GLOB (repeatable)")
    summarize.add_argument("--no-seed-groups", action="store_true",
                           help="do not group runs that differ only in their seed")
    summarize.add_argument("--seed-key", action="append", metavar="KEY",
                           help="a seed: a config path, or component:path inside that "
                                "component's entry (repeatable; default: "
                                + ", ".join(DEFAULT_SEED_KEYS) + ")")
    summarize.add_argument("--ignore-key", action="append", default=[], metavar="PATH",
                           help="a config path (and everything under it) not to compare (repeatable; "
                                "added to the defaults: " + ", ".join(DEFAULT_IGNORED_KEYS) + ")")
    summarize.add_argument("--no-default-ignores", action="store_true",
                           help="compare the default ignored keys too")
    summarize.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    summarize.add_argument("--table", choices=TABLES,
                           help="with --format csv: which table (default: groups, or runs "
                                "with --no-seed-groups)")
    summarize.add_argument("--output", "-o", metavar="PATH",
                           help="write the report here (default: standard output)")
    summarize.add_argument("--strict", action="store_true",
                           help="exit with status 1 when the report lists problems")
    return parser


def _summarize(args) -> int:
    if args.table is not None and args.format != "csv":
        raise ReportError("--table only applies with --format csv")
    if args.seed_key is not None and args.no_seed_groups:
        raise ReportError("--seed-key has no effect with --no-seed-groups")
    if args.table == "groups" and args.no_seed_groups:
        raise ReportError("there is no groups table with --no-seed-groups")
    discovery = discover_runs(*args.roots, include=args.include, exclude=args.exclude)
    if args.output is not None:
        output = os.path.abspath(args.output)
        for run in discovery.runs:
            if os.path.commonpath([output, run.session_dir]) == run.session_dir:
                raise ReportError(
                    f"--output {args.output} is inside run {run.run_id}; write the "
                    "report outside the run directories"
                )
    ignored = [] if args.no_default_ignores else list(DEFAULT_IGNORED_KEYS)
    summary = summarize_runs(
        discovery,
        args.metric,
        tags=args.tags,
        last_n=args.last_n,
        group_by_seed=not args.no_seed_groups,
        seed_keys=args.seed_key or DEFAULT_SEED_KEYS,
        ignored_keys=ignored + args.ignore_key,
        sort=args.sort,
    )
    if args.format == "markdown":
        text = render_markdown(summary)
    elif args.format == "json":
        text = render_json(summary)
    else:
        table = args.table or ("runs" if args.no_seed_groups else "groups")
        text = render_csv(summary, table=table)
    if args.output is None:
        sys.stdout.write(text)
    else:
        directory = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}: {len(summary.runs)} run(s), "
              f"{len(summary.problems)} problem(s)", file=sys.stderr)
    if summary.problems and args.strict:
        print(f"{len(summary.problems)} problem(s):", file=sys.stderr)
        for problem in summary.problems:
            print(f"- {problem}", file=sys.stderr)
        return EXIT_PROBLEMS
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _summarize(args)
    except ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
