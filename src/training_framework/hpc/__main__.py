"""`python -m training_framework.hpc <command>`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
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


class HashProgress:
    """Reports hashing of an upload source on stderr once it takes longer
    than `delay` seconds: a line rewritten in place on a terminal, a line
    every `log_interval` seconds otherwise, and a closing summary."""

    def __init__(self, stream=None, delay: float = 1.0, log_interval: float = 15.0,
                 clock=time.monotonic):
        self.stream = stream or sys.stderr
        self.delay = delay
        self.log_interval = log_interval
        self.clock = clock
        self._root = None

    def __call__(self, root: str, done: int, total: int, size: int,
                 hashed: int | None = None) -> None:
        now = self.clock()
        if root != self._root:
            self._root, self._started, self._last, self._shown = root, now, None, False
        finished = done == total
        if now - self._started < self.delay and not (finished and self._shown):
            return
        tty = self.stream.isatty()
        interval = 0.2 if tty else self.log_interval
        if not finished and self._last is not None and now - self._last < interval:
            return
        self._last, self._shown = now, True
        # A run over cached files only compares timestamps; say so, or it
        # looks like the whole dataset is hashed again.
        verb = "hashing" if hashed is None else "checking"
        line = (f"[hpc] {verb} {root} (upload source): {done:,}/{total:,} files, "
                f"{_size(size)}")
        if hashed is not None:
            line += f" ({hashed:,} hashed, {done - hashed:,} unchanged since the last run)"
        if finished:
            line += f", done in {now - self._started:.0f}s"
        if tty:
            self.stream.write("\r\033[K" + line + ("\n" if finished else ""))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()


class WorkstationFacts:
    """`LocalFacts` backed by this machine's filesystem and hash cache."""

    def __init__(self, repository_root: Path, cache_path: Path | None = None,
                 progress=None):
        self.repository_root = str(repository_root)
        self.home = os.path.expanduser("~")
        self._cache_path = cache_path or default_cache_path()
        self._cache = None
        self._progress = progress if progress is not None else HashProgress()
        self._summaries = {}

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def content(self, path: str):
        if path in self._summaries:       # several sources can name one dataset
            return self._summaries[path]
        if self._cache is None:
            self._cache = HashCache(self._cache_path)
        start = self._cache.hashed

        def report(root, done, total, size):
            self._progress(root, done, total, size, self._cache.hashed - start)

        summary = content_summary(path, self._cache, report)
        self._summaries[path] = summary.key, summary.files, summary.size
        return self._summaries[path]


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
    status = commands.add_parser("status", help="show the state of every task of a submitted batch")
    status.add_argument("batch_id", help="a batch id, or the path to its receipt")
    status.add_argument("--receipts", type=Path, help="receipt directory "
                        "(default: $XDG_STATE_HOME/training-framework/submissions)")
    profile_options(status)
    listing = commands.add_parser(
        "batches", help="list every batch you submitted, past and present",
        description="List every batch submitted from this machine (from its receipts). "
                    "--live adds what each batch is doing now; --tasks shows every task.")
    listing.add_argument("--live", action="store_true",
                         help="connect to the cluster(s) and show what each batch is doing now")
    listing.add_argument("--tasks", action="store_true",
                         help="show every task of each batch (implies --live)")
    listing.add_argument("--remote", action="store_true",
                         help="also list batches submitted from other machines (needs a profile)")
    listing.add_argument("--active", action="store_true",
                         help="only batches with queued or running tasks (implies --live)")
    listing.add_argument("--since", metavar="AGE", help="only batches submitted within AGE: 30m, 12h, 7d, 2w")
    listing.add_argument("--name", help="only batches whose id contains NAME")
    listing.add_argument("--limit", type=int, help="only the newest LIMIT batches")
    listing.add_argument("--json", action="store_true", help="print JSON instead of a table")
    listing.add_argument("--receipts", type=Path, help="receipt directory "
                         "(default: $XDG_STATE_HOME/training-framework/submissions)")
    profile_options(listing)
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
    from training_framework.hpc.scheduler import get_scheduler
    from training_framework.hpc.services.artifacts import WorkstationProbe

    executor = SshExecutor(profile.ssh_host)
    scheduler = get_scheduler(profile.scheduler)
    return {
        "executor": executor,
        "filesystem": SshFilesystem(executor),
        # A terminal shows rsync's own live line; elsewhere (an IDE console,
        # a log) a timestamped line every 30s, or a long upload looks hung.
        "transfer": (RsyncTransfer(profile.ssh_host, show_progress=True) if sys.stderr.isatty()
                     else RsyncTransfer(profile.ssh_host, report=_progress)),
        "submitter": scheduler.submitter(executor),
        "probe": WorkstationProbe(scheduler.owner_probe(executor), LocalProcessProbe()),
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


def _progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _require_tools(*names: str) -> None:
    """Fail before any slow work (hashing, staging) when a tool is missing."""
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise HpcError(
            f"{' and '.join(missing)} not found on this machine's PATH; submit uses "
            "ssh to reach the cluster and rsync to copy the source and data "
            "(install them, e.g. `apt install openssh-client rsync` or "
            "`conda install -c conda-forge rsync`)"
        )


def _submit(args, profile) -> int:
    from training_framework.hpc.remote.runtime import current_process_owner
    from training_framework.hpc.services.staging import SourceStager
    from training_framework.hpc.services.submit import submit

    _require_tools("ssh", "rsync")
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
            upload_limit=profile.upload_limit, review=_review(args.yes), progress=_progress,
            receipt_info={
                "ssh_host": profile.ssh_host,
                "repository": str(repository.root),
                "manifest": Path(args.batch).resolve().relative_to(
                    repository.root.resolve()).as_posix(),
                "submitted_from": socket.gethostname(),
            },
        )
    except HpcError as error:
        receipt = getattr(error, "receipt", None)
        if receipt is not None:
            print(f"{receipt.status}: {plan.batch_dir}")
            print(f"receipt: {receipt.path}")
        raise
    except KeyboardInterrupt as interrupt:
        receipt = getattr(interrupt, "receipt", None)
        print("\ninterrupted", file=sys.stderr)
        if receipt is not None:
            print(f"{receipt.status}: {plan.batch_dir}")
            if receipt.data.get("error"):
                print(f"  {receipt.data['error']}")
            print(f"receipt: {receipt.path}")
        return 130
    print(f"{receipt.status}: {plan.batch_dir}")
    for node_id, job_ref in receipt.submitted_jobs():
        print(f"  {node_id}: job {job_ref}")
    print(f"receipt: {receipt.path}")
    return 0 if receipt.status == "submitted" else 1


def _status(args, profile) -> int:
    from training_framework.hpc.scheduler import get_scheduler
    from training_framework.hpc.services.receipt import default_receipts_dir
    from training_framework.hpc.services.status import batch_status, format_rows

    path = Path(args.batch_id)
    if not path.is_file():
        path = Path(args.receipts or default_receipts_dir()) / f"{args.batch_id}.json"
    if not path.is_file():
        raise HpcError(f"no receipt for {args.batch_id} (looked at {path})")
    receipt = json.loads(path.read_text())
    ports = _cluster(profile)
    scheduler = get_scheduler(profile.scheduler)
    job_states = scheduler.job_states(ports["executor"]) if scheduler.job_states else None
    print(f"batch {receipt['batch_id']}: {receipt['status']} | {receipt['batch_dir']}")
    rows = batch_status(receipt, ports["filesystem"], ports["executor"], job_states)
    print(format_rows(rows))
    return 0


def _batches(args) -> int:
    from training_framework.hpc.ports.ssh import SshExecutor
    from training_framework.hpc.scheduler import get_scheduler
    from training_framework.hpc.services import batches as listing
    from training_framework.hpc.services.receipt import default_receipts_dir

    live = args.live or args.tasks or args.active
    since = None
    if args.since:
        try:
            since = listing.parse_age(args.since)
        except ValueError as error:
            raise HpcError(str(error)) from None
    # The profile is optional: receipts name their cluster. It is needed for
    # --remote, and for receipts written before they recorded the cluster.
    profile = None
    if args.remote or args.profile or args.profiles or default_profiles_path().exists():
        profile = _profile(args)
    if live or args.remote:
        _require_tools("ssh")
    views = listing.load_receipts(str(args.receipts or default_receipts_dir()))

    def connect(ssh_host: str, scheduler: str):
        executor = SshExecutor(ssh_host)
        bundle = get_scheduler(scheduler)
        return executor, (lambda: bundle.user_job_states(executor)) if bundle.user_job_states else None

    if args.remote:
        if profile is None:
            raise HpcError("--remote needs a profile (it lists the profile's roots.batches)")
        known = {view.batch_id for view in views}
        executor = SshExecutor(profile.ssh_host)
        for receipt in listing.remote_receipts(executor, profile.roots["batches"]):
            if receipt["batch_id"] not in known:
                views.append(listing.BatchView(receipt, "cluster"))
        views = listing.sort_views(views)
    # Cheap filters first, so --live only reads the batches that are shown.
    views = listing.select(views, since=since, name=args.name,
                           limit=None if args.active else args.limit)
    if live:
        fallback = (profile.ssh_host, profile.scheduler) if profile else None
        listing.add_live_state(views, connect, fallback)
        if args.active:
            views = listing.select(views, active=True, limit=args.limit)
    if args.json:
        print(json.dumps(listing.as_json(views, tasks=args.tasks), indent=2))
    else:
        print(listing.format_views(views, live=live, tasks=args.tasks, origins=args.remote))
    return 0


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
        if args.command == "batches":
            return _batches(args)
        profile = _profile(args)
        if args.command in ("probe-python", "status"):
            _require_tools("ssh")
        if args.command == "probe-python":
            return _probe_python(profile)
        if args.command == "submit":
            return _submit(args, profile)
        if args.command == "status":
            return _status(args, profile)
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
