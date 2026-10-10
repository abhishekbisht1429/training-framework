"""The `setup` node's work (building block B10), before any environment
exists: lease, build, install, `pip check`, preflight, record, marker.

Never imported or run from the package: `bootstrap.render_bootstrap`
bundles it with `runtime`, `lease`, and `preflight` into one standard-library
file (Python 3.8 syntax) and defines `PROTOCOL_VERSION`. Imports from this
package must stay on single lines; the bundler removes them.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from training_framework.hpc.remote.runtime import OwnerProbeRouter, ProcessProbe, SlurmJobProbe, StdlibFilesystem, current_owner
from training_framework.hpc.remote.lease import Lease, check_marker, write_marker
from training_framework.hpc.remote.preflight import compare


class SetupFailed(Exception):
    pass


def _run(argv, what, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise SetupFailed("{} failed (exit {}):\n{}".format(
            what, result.returncode, (result.stdout + result.stderr)[-4000:]))
    return result


def build_environment(environment, source_dir):
    location = environment["location"]
    if environment["backend"] == "conda":
        _run(["conda", "create", "--yes", "--quiet", "-p", location, "-c", "conda-forge",
              "--override-channels", "python={}".format(environment["python"])]
             + list(environment["conda_packages"]), "conda create")
    else:
        _run([sys.executable, "-m", "venv", location], "venv")
    python = os.path.join(location, "bin", "python")
    requirements = os.path.join(source_dir, environment["requirements"])
    _run([python, "-m", "pip", "install", "--no-deps", "--no-input",
          "--disable-pip-version-check", "-r", requirements], "pip install")
    _run([python, "-m", "pip", "check", "--disable-pip-version-check"], "pip check")
    return python


def run_preflight(plan, plan_path, python, source_dir):
    script = plan["environment"]["preflight_script"]
    result = _run(["bash", "-c", script, "bash", python, "-m", "training_framework.hpc.task",
                   "--preflight", plan_path], "preflight")
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    problems = compare(facts, plan["preflight"], source_dir)
    if problems:
        raise SetupFailed("preflight failed:\n" + "\n".join("- " + p for p in problems))
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bootstrap.py")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    with open(args.plan) as handle:
        plan = json.load(handle)
    environment = plan["environment"]
    location, key = environment["location"], environment["key"]
    batch_dir = plan["batch_dir"]
    source_dir = batch_dir + "/source"
    filesystem = StdlibFilesystem()
    started = time.time()
    record = {
        "node": "setup", "batch_id": plan["batch_id"], "sha": plan["sha"],
        "execution_protocol_version": plan["protocol_version"],
        "environment_key": key, "location": location, "owner": current_owner(),
        "status": None, "error": None, "facts": None,
    }
    status = 1
    try:
        if plan["protocol_version"] != PROTOCOL_VERSION:  # noqa: F821 (set by the bundler)
            raise SetupFailed("this bootstrap speaks execution protocol {} but the plan "
                              "uses {}".format(PROTOCOL_VERSION, plan["protocol_version"]))  # noqa: F821
        if check_marker(filesystem, location, key):
            record["status"] = "already ready"
        else:
            lease = Lease(filesystem, OwnerProbeRouter(SlurmJobProbe(), ProcessProbe()),
                          location, record["owner"])
            lease.acquire()
            try:
                if check_marker(filesystem, location, key):
                    record["status"] = "made ready meanwhile"
                else:
                    if filesystem.exists(location):
                        filesystem.delete(location)   # a dead builder's partial environment
                    python = build_environment(environment, source_dir)
                    record["facts"] = run_preflight(plan, args.plan, python, source_dir)
                    write_marker(filesystem, location, key)
                    record["status"] = "built"
            finally:
                lease.release()
        status = 0
    except Exception as error:   # recorded, then reported by the exit status
        record["status"] = "failed"
        record["error"] = "{}: {}".format(type(error).__name__, error)
        sys.stderr.write(record["error"] + "\n")
    record["duration_seconds"] = round(time.time() - started, 3)
    filesystem.write(batch_dir + "/nodes/setup/record.json",
                     (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    return status
