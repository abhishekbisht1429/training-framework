"""The remote entry point: `python -m training_framework.hpc.task`.

- `--config CFG --record REC --plan PLAN [engine options]`: run one task
  (the README entry point), then write its record.
- `--preflight PLAN`: print the facts preflight compares (B5) as JSON.

Runs inside the batch environment, so it may import the framework.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import importlib.metadata
import json
import os
import signal
import socket
import sys
import tempfile
import traceback
from datetime import datetime, timezone

import yaml

from training_framework.hpc import EXECUTION_PROTOCOL_VERSION
from training_framework.hpc.remote.runtime import interpreter_identity

SUPPORTED_PROTOCOL_VERSIONS = (EXECUTION_PROTOCOL_VERSION,)
INTERRUPTED_EXIT = 75
#: What this remote side can do beyond the protocol; `resume` checks for it.
CAPABILITIES = ("resume",)


def interpreter_facts() -> dict:
    return interpreter_identity()


def framework_facts() -> dict:
    import training_framework

    try:
        distribution = importlib.metadata.version("training-framework")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    return {"file": os.path.realpath(training_framework.__file__), "distribution": distribution}


def environment_hash() -> str:
    """Hash of the installed distributions (what `pip freeze` would list)."""
    entries = sorted({
        f"{(dist.metadata['Name'] or '').lower()}=={dist.version}"
        for dist in importlib.metadata.distributions()
    })
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def preflight_facts(plan: dict) -> dict:
    components = {}
    for package in plan["preflight"]["components_packages"]:
        try:
            importlib.import_module(package)
            components[package] = None
        except Exception as error:  # reported, not raised
            components[package] = f"{type(error).__name__}: {error}"
    return {
        "protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        "capabilities": list(CAPABILITIES),
        "interpreter": interpreter_facts(),
        "framework": framework_facts(),
        "components": components,
        "cluster_paths": {
            path: os.path.exists(path) for path in plan["preflight"]["cluster_paths"]
        },
    }


def _write_atomic(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    with os.fdopen(descriptor, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _provenance(plan: dict) -> dict:
    environment_artifact = next(
        (a for a in plan["artifacts"] if a["kind"] == "environment"), None,
    )
    return {
        "batch_id": plan["batch_id"],
        "execution_protocol_version": plan["protocol_version"],
        "sha": plan["sha"],
        "manifest_hash": plan["manifest_hash"],
        "code_mode": plan["code_mode"],
        "environment_key": plan["environment"]["key"],
        "requirements_hash": environment_artifact["details"]["requirements"] if environment_artifact else None,
        "interpreter": interpreter_facts(),
        "framework_distribution": framework_facts()["distribution"],
        "environment_hash": environment_hash(),
    }


def _run_directories(config: dict) -> list[str]:
    sessions_dir = config["sessions"][0]["session_config"]["sessions_dir"]
    return sorted(glob.glob(os.path.join(sessions_dir, "session_*")))


def log(message: str) -> None:
    """A timestamped line in the task's scheduler log."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[hpc {stamp}] {message}", flush=True)


def _variant_text(config_path: str) -> str:
    path = os.path.join(os.path.dirname(config_path), "variant.json")
    try:
        with open(path) as handle:
            variant = json.load(handle)
    except (OSError, ValueError):
        return ""
    return " ".join(
        f"{'.'.join(item['address']['path'])}={json.dumps(item['value'])}" for item in variant
    )


def _previous_attempts(record_path: str) -> list[dict]:
    """The records of earlier attempts: the current record, and its history."""
    try:
        with open(record_path) as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        return []
    history = previous.pop("previous_attempts", [])
    previous.setdefault("attempt", 1)
    return [*history, previous]


def run_task(config_path: str, record_path: str, plan_path: str,
             engine_options: list[str], resume: str | None = None, attempt: int = 1) -> int:
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        signal.default_int_handler(signum, frame)

    # A shell starts background jobs with SIGINT ignored, and Python then
    # never installs its KeyboardInterrupt handler; the wrapper forwards the
    # scheduler's shutdown signal as SIGINT, so it must be handled here.
    signal.signal(signal.SIGINT, on_sigint)

    with open(plan_path) as handle:
        plan = json.load(handle)
    with open(config_path, "rb") as handle:
        config_bytes = handle.read()
    config = yaml.safe_load(config_bytes)
    task_name = os.path.relpath(os.path.dirname(config_path), os.path.join(plan["batch_dir"], "tasks"))
    log(f"task {task_name} | {_variant_text(config_path) or 'no matrix values'}")
    log(f"batch {plan['batch_id']} | commit {plan['sha'][:12]} | host {socket.gethostname()} | "
        f"GPUs {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log(f"config {config_path}")
    if resume:
        log(f"attempt {attempt}: resuming from {resume}")
    started = datetime.now(timezone.utc).isoformat()
    status, error = 0, None
    try:
        from training_framework.engine import Configurator, TrainingEngine

        if resume:
            # The checkpoint holds the session (config, iteration, RNG, data
            # position); only the launch topology comes from this run.
            sys.argv = [sys.argv[0], "--resume-session", resume, *engine_options]
        else:
            sys.argv = [sys.argv[0], "--config", config_path, *engine_options]
        with TrainingEngine(Configurator()) as engine:
            engine.start_session()
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as caught:  # recorded, then reported by the exit status
        status = 1
        error = "".join(traceback.format_exception(caught))
        traceback.print_exc()
    record = {
        **_provenance(plan),
        "task_config": config_path,
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "run_directories": _run_directories(config),
        "interrupted": interrupted,
        "exit_status": INTERRUPTED_EXIT if interrupted else status,
        "error": error,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
    }
    if resume:
        record["resumed_from"] = resume
        record["previous_attempts"] = _previous_attempts(record_path)
    _write_atomic(record_path, record)
    outcome = "interrupted" if interrupted else ("ok" if status == 0 else "failed")
    log(f"finished: exit {record['exit_status']} ({outcome}) | record {record_path}")
    return record["exit_status"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training_framework.hpc.task")
    parser.add_argument("--preflight", metavar="PLAN")
    parser.add_argument("--prepare", metavar="NODE")
    parser.add_argument("--config")
    parser.add_argument("--record")
    parser.add_argument("--plan")
    parser.add_argument("--resume", metavar="CHECKPOINT",
                        help="continue the task from this checkpoint (a later attempt)")
    parser.add_argument("--attempt", type=int, default=1)
    args, engine_options = parser.parse_known_args(argv)
    if args.preflight:
        with open(args.preflight) as handle:
            plan = json.load(handle)
        print(json.dumps(preflight_facts(plan), sort_keys=True))
        return 0
    if args.prepare:
        if not args.plan:
            parser.error("--prepare needs --plan")
        from training_framework.hpc.remote.prepare import run_prepare

        return run_prepare(args.plan, args.prepare)
    if not (args.config and args.record and args.plan):
        parser.error("a task needs --config, --record, and --plan")
    return run_task(args.config, args.record, args.plan, engine_options,
                    resume=args.resume, attempt=args.attempt)


if __name__ == "__main__":
    sys.exit(main())
