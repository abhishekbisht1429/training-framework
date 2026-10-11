"""A `prepare` node's work on the cluster (`hpc.task --prepare NODE --plan P`):
build or download one data path under the B3 lease.

Success means exit status 0 and the path exists; only then is the ready
marker written. On takeover the command simply runs again (it must be safe to
re-run); user data is never deleted.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from training_framework.hpc.compile.data import expand_command
from training_framework.hpc.remote.lease import Lease, check_marker, write_marker
from training_framework.hpc.remote.runtime import StdlibFilesystem, current_owner, owner_probe


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[hpc prepare {stamp}] {message}", flush=True)


def run_prepare(plan_path: str, node_id: str) -> int:
    with open(plan_path) as handle:
        plan = json.load(handle)
    node = next(node for node in plan["nodes"] if node["id"] == node_id)
    artifact = next(a for a in plan["artifacts"]
                    if a["kind"] == "prepare" and a["key"] == node["artifact"])
    location, key = artifact["location"], artifact["key"]
    filesystem = StdlibFilesystem()
    record = {"node": node_id, "batch_id": plan["batch_id"], "sha": plan["sha"],
              "key": key, "location": location, "owner": current_owner(plan["scheduler"]),
              "status": None, "error": None}
    started = time.time()
    status = 1
    log(f"{node_id}: data source '{artifact['source']}' at {location}")
    try:
        if check_marker(filesystem, location, key):
            record["status"] = "already ready"
            log("already ready; nothing to do")
        else:
            lease = Lease(filesystem, owner_probe(plan["scheduler"]),
                          location, record["owner"], notify=log)
            lease.acquire()
            try:
                if check_marker(filesystem, location, key):
                    record["status"] = "made ready meanwhile"
                    log("another job prepared it meanwhile")
                else:
                    command = expand_command(artifact["details"]["command"], location)
                    log("running: " + " ".join(command))
                    result = subprocess.run(command, cwd=f"{plan['batch_dir']}/source")
                    if result.returncode != 0:
                        raise RuntimeError(f"command exited {result.returncode}: {command}")
                    if not os.path.exists(location):
                        raise RuntimeError(f"command succeeded but did not create {location}")
                    write_marker(filesystem, location, key)
                    record["status"] = "prepared"
                    log("prepared")
            finally:
                lease.release()
        status = 0
    except Exception as error:   # recorded, then reported by the exit status
        record["status"], record["error"] = "failed", f"{type(error).__name__}: {error}"
        log("FAILED: " + record["error"])
    record["duration_seconds"] = round(time.time() - started, 3)
    filesystem.write(f"{plan['batch_dir']}/nodes/{node_id}/record.json",
                     (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    return status
