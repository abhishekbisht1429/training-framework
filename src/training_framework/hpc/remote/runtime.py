"""Standard-library-only pieces shared by the workstation and the bootstrap.

The bootstrap (B10) runs before any environment exists, so it bundles this
module verbatim; the controller imports the same code (`local.py`), so there
is one implementation. Python 3.8 syntax, standard library only, no imports
from this package. Owners are plain dicts here: {"job_ref": ...} or
{"host": ..., "pid": ..., "start_time": ...}.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile


def interpreter_identity():
    """This interpreter's identity. Inside a venv, `executable` is the base
    interpreter the venv was made from (resolved), so an environment's
    identity equals that of the site interpreter it was built from."""
    executable = sys.executable
    if sys.prefix != sys.base_prefix:
        executable = getattr(sys, "_base_executable", executable)
    return {
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "executable": os.path.realpath(executable),
        "platform": "{}-{}".format(sys.platform, platform.machine()),
    }


class StdlibFilesystem:
    """`RemoteFilesystem` on the local filesystem."""

    def mkdir_atomic(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            return False
        return True

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def rename(self, source, target):
        os.replace(source, target)

    def read(self, path):
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    def write(self, path, data):
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def delete(self, path):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.unlink(path)

    def exists(self, path):
        return os.path.lexists(path)


def process_start_time(pid):
    """The kernel's start time of `pid` (clock ticks since boot), or None."""
    try:
        with open("/proc/{}/stat".format(pid)) as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        return fields[19]   # field 22 overall; the 20th after "(comm)"
    except (OSError, IndexError):
        return None


def current_process_owner():
    pid = os.getpid()
    return {"host": socket.gethostname(), "pid": pid,
            "start_time": process_start_time(pid)}


def current_owner():
    """The owner to record for this process: its Slurm job when it runs in
    one (checkable from any host), else the process itself."""
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        return {"job_ref": job}
    return current_process_owner()


class ProcessProbe:
    """Liveness of process owners on this host; None for anything else."""

    def __init__(self, host=None):
        self.host = host or socket.gethostname()

    def is_alive(self, owner):
        if owner.get("job_ref") is not None or owner.get("host") != self.host:
            return None
        pid = owner.get("pid")
        if pid is None:
            return None
        start = process_start_time(pid)
        return start is not None and start == owner.get("start_time")


class SlurmJobProbe:
    """Liveness of Slurm job owners, via `squeue` (any host)."""

    def __init__(self, run=None):
        self._run = run or (lambda argv: subprocess.run(
            argv, capture_output=True, text=True, check=False))

    def is_alive(self, owner):
        job = owner.get("job_ref")
        if job is None:
            return None
        result = self._run(["squeue", "--noheader", "--jobs", str(job), "--format", "%T"])
        if result.returncode != 0:
            if "Invalid job id" in (result.stderr or ""):
                return False
            raise RuntimeError("squeue failed: {}".format((result.stderr or "").strip()))
        return bool(result.stdout.strip())


class OwnerProbeRouter:
    """Routes an owner to the probe for its kind."""

    def __init__(self, job_probe, process_probe):
        self.job_probe = job_probe
        self.process_probe = process_probe

    def is_alive(self, owner):
        if owner.get("job_ref") is not None:
            return self.job_probe.is_alive(owner)
        return self.process_probe.is_alive(owner)
