"""`python -m training_framework.hpc <command>`."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training_framework.hpc.compile.content import HashCache, content_summary, default_cache_path
from training_framework.hpc.config.schema import parse_profiles
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.repository import GitRepository


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

    def profile_options(command):
        command.add_argument("--profile", help="profile name (default: the file's 'default')")
        command.add_argument("--profiles", type=Path,
                             help="profile file (default: $XDG_CONFIG_HOME/training-framework/hpc.yaml)")

    for name, help_text in (
            ("validate", "compile the batch and report every offline problem"),
            ("render", "write a preview of the batch directory"),
            ("submit", "stage the batch on the cluster and submit its jobs"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("batch", help="path to the committed batch manifest")
        profile_options(command)
        if name == "validate":
            command.add_argument("--suggest-data", action="store_true",
                                 help="print data.sources entries for uncovered local paths")
        elif name == "render":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--fresh-env", action="store_true",
                                 help="build a managed environment for this batch only")
            command.add_argument("--yes", action="store_true",
                                 help="do not ask (also accepts uploads above upload_limit)")
    probe = commands.add_parser("probe-python",
                                help="print the site interpreter's identity for the profile")
    profile_options(probe)
    return parser


def _profile(args):
    path = args.profiles or default_profiles_path()
    if not path.exists():
        raise HpcError(f"profile file {path} does not exist")
    return parse_profiles(yaml.safe_load(path.read_text()), args.profile, str(path))


def _compile(args, profile, fresh_env=False):
    # Imported here so `--help` stays fast.
    from training_framework.hpc.compile.compiler import compile_plan

    batch = Path(args.batch).resolve()
    repository = GitRepository(batch.parent)
    manifest_path = batch.relative_to(repository.root.resolve()).as_posix()
    compiled = compile_plan(
        manifest_path=manifest_path,
        repository=repository,
        profile=profile,
        facts=WorkstationFacts(repository.root),
        clock=datetime.now(timezone.utc),
        fresh_env=fresh_env,
    )
    return compiled, repository


def _cluster(profile):
    from training_framework.hpc.ports.local import LocalProcessProbe
    from training_framework.hpc.ports.ssh import RsyncTransfer, SshExecutor, SshFilesystem
    from training_framework.hpc.scheduler.slurm_submitter import RemoteOwnerProbe, SlurmSubmitter
    from training_framework.hpc.services.artifacts import WorkstationProbe

    executor = SshExecutor(profile.ssh_host)
    return {
        "executor": executor,
        "filesystem": SshFilesystem(executor),
        "transfer": RsyncTransfer(profile.ssh_host),
        "submitter": SlurmSubmitter(executor),
        "probe": WorkstationProbe(RemoteOwnerProbe(executor), LocalProcessProbe()),
    }


def _size(value) -> str:
    value = float(value or 0)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _review(assume_yes: bool):
    def review(plan, summary) -> bool:
        print(f"batch {plan.batch_id}: {summary.tasks} task(s) in {summary.groups} group(s)")
        print(f"environment: {'ready' if summary.environment_ready else 'will be built'}")
        for source, local, size in summary.uploads:
            print(f"upload {source}: {local} ({_size(size)})")
        if summary.uploads:
            limit = f" (limit {_size(summary.upload_limit)})" if summary.upload_limit else ""
            print(f"total upload: {_size(summary.upload_bytes)}{limit}")
        if assume_yes:
            return True
        if not sys.stdin.isatty():
            print("error: not interactive; pass --yes to submit", file=sys.stderr)
            return False
        prompt = ("Upload exceeds upload_limit. Submit anyway? [y/N] " if summary.over_limit
                  else "Submit? [y/N] ")
        return input(prompt).strip().lower() in ("y", "yes")
    return review


def _submit(args, profile) -> int:
    from training_framework.hpc.remote.runtime import current_process_owner
    from training_framework.hpc.services.staging import SourceStager
    from training_framework.hpc.services.submit import submit

    compiled, repository = _compile(args, profile, fresh_env=args.fresh_env)
    plan = compiled.plan
    for note in plan.notes:
        print(f"note: {note}")
    ports = _cluster(profile)
    manifest = compiled.manifest
    stager = SourceStager(repository, ports["executor"], ports["transfer"],
                          manifest.source_strategy, manifest.source_url)
    try:
        receipt = submit(
            plan, manifest.raw, stager=stager, filesystem=ports["filesystem"],
            executor=ports["executor"], submitter=ports["submitter"],
            transfer=ports["transfer"], probe=ports["probe"], owner=current_process_owner(),
            upload_limit=profile.upload_limit, review=_review(args.yes),
        )
    except HpcError as error:
        receipt = getattr(error, "receipt", None)
        if receipt is not None:
            print(f"{receipt.status}: {plan.batch_dir}")
            print(f"receipt: {receipt.path}")
        raise
    print(f"{receipt.status}: {plan.batch_dir}")
    for node_id, job_ref in receipt.submitted_jobs():
        print(f"  {node_id}: job {job_ref}")
    print(f"receipt: {receipt.path}")
    return 0 if receipt.status == "submitted" else 1


def _probe_python(profile) -> int:
    from training_framework.hpc.compile.compiler import (
        SET_VARIABLES,
        UNSET_VARIABLES,
        bootstrap_interpreter,
    )
    from training_framework.hpc.compile.plan import ExecutionContext
    from training_framework.hpc.services.artifacts import identity_block, probe_identity

    if profile.env.python_source != "site":
        raise HpcError("probe-python reports a 'site' interpreter; uv and conda "
                       "identities come from environment.python")
    interpreter, _ = bootstrap_interpreter(profile.env, None, ".")
    context = ExecutionContext(
        phase="setup", interpreter=interpreter, environment=None, working_directory=".",
        pythonpath=(), set_variables=SET_VARIABLES, unset_variables=UNSET_VARIABLES,
        hooks=tuple(profile.hooks["common"]) + tuple(profile.hooks["setup"]),
    )
    print(identity_block(probe_identity(_cluster(profile)["executor"], context)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = _profile(args)
        if args.command == "probe-python":
            return _probe_python(profile)
        if args.command == "submit":
            return _submit(args, profile)
        compiled, _ = _compile(args, profile)
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
