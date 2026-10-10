"""`python -m training_framework.hpc <command>`."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training_framework.hpc.compile.content import HashCache, content_summary, default_cache_path
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.repository import GitRepository
from training_framework.hpc.config.schema import parse_profiles


def default_profiles_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "training-framework" / "hpc.yaml"


class WorkstationFacts:
    """`LocalFacts` backed by this machine's filesystem and hash cache."""

    def __init__(self, repository_root: Path, cache_path: Path | None = None):
        self.repository_root = str(repository_root)
        self.home = os.path.expanduser("~")
        self._cache_path = cache_path or default_cache_path()
        self._cache = None

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def content(self, path: str):
        if self._cache is None:
            self._cache = HashCache(self._cache_path)
        summary = content_summary(path, self._cache)
        return summary.key, summary.files, summary.size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m training_framework.hpc")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
            ("validate", "compile the batch and report every offline problem"),
            ("render", "write a preview of the batch directory"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("batch", help="path to the committed batch manifest")
        command.add_argument("--profile", help="profile name (default: the file's 'default')")
        command.add_argument("--profiles", type=Path, help="profile file (default: $XDG_CONFIG_HOME/training-framework/hpc.yaml)")
        if name == "validate":
            command.add_argument("--suggest-data", action="store_true",
                                 help="print data.sources entries for uncovered local paths")
        else:
            command.add_argument("--output", type=Path, required=True)
    return parser


def _compile(args):
    # Imported here so `--help` stays fast.
    from training_framework.hpc.compile.compiler import compile_plan

    profiles_path = args.profiles or default_profiles_path()
    if not profiles_path.exists():
        raise HpcError(f"profile file {profiles_path} does not exist")
    profile = parse_profiles(
        yaml.safe_load(profiles_path.read_text()), args.profile, str(profiles_path),
    )
    batch = Path(args.batch).resolve()
    repository = GitRepository(batch.parent)
    manifest_path = batch.relative_to(repository.root.resolve()).as_posix()
    return compile_plan(
        manifest_path=manifest_path,
        repository=repository,
        profile=profile,
        facts=WorkstationFacts(repository.root),
        clock=datetime.now(timezone.utc),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        compiled = _compile(args)
    except HpcError as error:  # includes PlanError, with every problem
        print(f"error: {error}", file=sys.stderr)
        return 2
    plan = compiled.plan
    for note in plan.notes:
        print(f"note: {note}")
    if args.command == "validate":
        print(f"ok: {len(plan.tasks)} task(s) in {sum(n.kind == 'task_set' for n in plan.nodes)} "
              f"group(s); commit {plan.sha[:12]}; environment "
              f"{plan.environment['key'] or plan.environment['location']}")
        if args.suggest_data and compiled.suggestions:
            print(yaml.safe_dump({"data": {"sources": compiled.suggestions}}, sort_keys=False))
        return 0
    from training_framework.hpc.compile.render import write_plan

    try:
        written = write_plan(plan, compiled.manifest.raw, args.output)
    except HpcError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(written)} file(s) to {args.output} (preview of {plan.batch_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
