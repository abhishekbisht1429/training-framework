"""A stub `training_framework` distribution for bootstrap tests.

Torch cannot be installed offline, so the environment a test bootstrap builds
gets this stub instead: its `hpc.task --preflight` reports real facts about
the interpreter and imports, like the real one (which phase-2 end-to-end
tests exercise with the real framework).
"""

from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path

STUB_TASK = r'''
import importlib, importlib.metadata, json, os, platform, sys

def interpreter():
    executable = sys.executable
    if sys.prefix != sys.base_prefix:
        executable = getattr(sys, "_base_executable", executable)
    return {"implementation": sys.implementation.name, "version": platform.python_version(),
            "executable": os.path.realpath(executable),
            "platform": "%s-%s" % (sys.platform, platform.machine())}

if __name__ == "__main__":
    plan = json.load(open(sys.argv[sys.argv.index("--preflight") + 1]))
    import training_framework
    try:
        distribution = importlib.metadata.version("training-framework")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    components = {}
    for package in plan["preflight"]["components_packages"]:
        try:
            importlib.import_module(package)
            components[package] = None
        except Exception as error:
            components[package] = repr(error)
    print(json.dumps({
        "protocol_versions": json.loads(os.environ.get("STUB_PROTOCOLS", "[1]")),
        "interpreter": interpreter(),
        "framework": {"file": os.path.realpath(training_framework.__file__),
                      "distribution": distribution},
        "components": components,
        "cluster_paths": {p: os.path.exists(p) for p in plan["preflight"]["cluster_paths"]},
    }))
'''

FILES = {
    "training_framework/__init__.py": "",
    "training_framework/hpc/__init__.py": "",
    "training_framework/hpc/task.py": STUB_TASK,
}


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build_wheel(directory: Path, name: str, version: str, files: dict[str, str]) -> Path:
    """A minimal, valid wheel; returns its path."""
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    contents = {path: text.encode() for path, text in files.items()}
    contents[f"{info}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    contents[f"{info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = "".join(f"{path},{_record_hash(data)},{len(data)}\n" for path, data in contents.items())
    contents[f"{info}/RECORD"] = (record + f"{info}/RECORD,,\n").encode()
    path = directory / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in contents.items():
            archive.writestr(member, data)
    return path


def pinned_line(name: str, wheel: Path) -> str:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return f"{name} @ file://{wheel}#sha256={digest}\n"
