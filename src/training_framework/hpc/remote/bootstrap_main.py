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

from training_framework.hpc.remote.runtime import StdlibFilesystem, current_owner, owner_probe
from training_framework.hpc.remote.lease import Lease, check_marker, write_marker
from training_framework.hpc.remote.preflight import compare


class SetupFailed(Exception):
    pass


def log(message):
    """A timestamped line in the setup job's log."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("[hpc setup {}] {}".format(stamp, message), flush=True)


def _run(argv, what, capture=False):
    """Run a step; its output goes straight to the log unless `capture`."""
    if capture:
        result = subprocess.run(argv, stdout=subprocess.PIPE, text=True)
    else:
        sys.stdout.flush()
        result = subprocess.run(argv)
    if result.returncode != 0:
        raise SetupFailed("{} failed (exit {}); its output is in this log".format(
            what, result.returncode))
    return result


def build_environment(environment, source_dir):
    location = environment["location"]
    log("creating the {} environment at {}".format(environment["backend"], location))
    if environment["backend"] == "conda":
        _run(["conda", "create", "--yes", "--quiet", "-p", location, "-c", "conda-forge",
              "--override-channels", "python={}".format(environment["python"])]
             + list(environment["conda_packages"]), "conda create")
    else:
        _run([sys.executable, "-m", "venv", location], "venv")
    python = os.path.join(location, "bin", "python")
    requirements = os.path.join(source_dir, environment["requirements"])
    log("installing the pinned requirements from {} (this can take several minutes)".format(
        environment["requirements"]))
    _run([python, "-m", "pip", "install", "--no-deps", "--no-input",
          "--disable-pip-version-check", "-r", requirements], "pip install")
    log("checking installed dependencies (pip check)")
    _run([python, "-m", "pip", "check", "--disable-pip-version-check"], "pip check")
    return python


def run_preflight(plan, plan_path, python, source_dir):
    log("running preflight in the new environment")
    script = plan["environment"]["preflight_script"]
    result = _run(["bash", "-c", script, "bash", python, "-m", "training_framework.hpc.task",
                   "--preflight", plan_path], "preflight", capture=True)
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    problems = compare(facts, plan["preflight"], source_dir)
    if problems:
        raise SetupFailed("preflight failed:\n" + "\n".join("- " + p for p in problems))
    log("preflight ok: Python {} | training_framework {}".format(
        facts["interpreter"]["version"], facts["framework"]["distribution"] or "from the checkout"))
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
        "environment_key": key, "location": location, "owner": current_owner(plan["scheduler"]),
        "status": None, "error": None, "facts": None,
    }
    status = 1
    try:
        if plan["protocol_version"] != PROTOCOL_VERSION:  # noqa: F821 (set by the bundler)
            raise SetupFailed("this bootstrap speaks execution protocol {} but the plan "
                              "uses {}".format(PROTOCOL_VERSION, plan["protocol_version"]))  # noqa: F821
        log("batch {} | environment {}".format(plan["batch_id"], key[:12]))
        if check_marker(filesystem, location, key):
            record["status"] = "already ready"
            log("the environment is already ready; nothing to do")
        else:
            lease = Lease(filesystem, owner_probe(plan["scheduler"]),
                          location, record["owner"], notify=log)
            lease.acquire()
            try:
                if check_marker(filesystem, location, key):
                    record["status"] = "made ready meanwhile"
                    log("another job finished building the environment meanwhile")
                else:
                    if filesystem.exists(location):
                        log("removing a partial environment left by a builder that died")
                        filesystem.delete(location)
                    python = build_environment(environment, source_dir)
                    record["facts"] = run_preflight(plan, args.plan, python, source_dir)
                    write_marker(filesystem, location, key)
                    record["status"] = "built"
                    log("environment ready")
            finally:
                lease.release()
        status = 0
    except Exception as error:   # recorded, then reported by the exit status
        record["status"] = "failed"
        record["error"] = "{}: {}".format(type(error).__name__, error)
        log("FAILED: " + record["error"])
    record["duration_seconds"] = round(time.time() - started, 3)
    filesystem.write(batch_dir + "/nodes/setup/record.json",
                     (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    return status
