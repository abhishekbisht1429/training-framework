"""Preflight comparison (building block B5): facts reported by the remote
side against expectations the compiler precomputed.

Standard library only, Python 3.8 syntax: the bootstrap bundles this module
and runs it before any environment (or `packaging`) exists. Every check is an
exact comparison; nothing here parses versions or specifiers.
"""

from typing import Dict, List


def _under(path, directory):
    directory = directory.rstrip("/") + "/"
    return path == directory[:-1] or path.startswith(directory)


def compare(facts: Dict, expected: Dict, source_dir: str) -> List[str]:
    problems = []  # type: List[str]

    version = expected.get("protocol_version")
    supported = facts.get("protocol_versions") or []
    if version not in supported:
        problems.append(
            "the remote training_framework.hpc supports execution protocol "
            "versions {} but this plan uses {}; install a framework version "
            "that supports it".format(supported or "none", version)
        )

    interpreter = facts.get("interpreter") or {}
    expectation = expected.get("interpreter") or {}
    identity = expectation.get("identity")
    prefix = expectation.get("version_prefix")
    if identity:
        for field in ("implementation", "version", "executable", "platform"):
            if interpreter.get(field) != identity.get(field):
                problems.append(
                    "interpreter {} is {!r}, expected {!r} (update the "
                    "profile's site_python_identity)".format(
                        field, interpreter.get(field), identity.get(field))
                )
    elif prefix:
        found = interpreter.get("version") or ""
        if found != prefix and not found.startswith(prefix + "."):
            problems.append(
                "interpreter version is {!r}, expected {}.x".format(found, prefix)
            )

    framework = facts.get("framework") or {}
    expected_framework = expected.get("framework") or {}
    location = framework.get("file") or ""
    distribution = framework.get("distribution")
    if expected_framework.get("location") == "checkout":
        if not _under(location, source_dir):
            problems.append(
                "training_framework is imported from {} instead of the "
                "checkout {}".format(location or "nowhere", source_dir)
            )
        if distribution is not None:
            problems.append(
                "a training-framework distribution ({}) is installed in the "
                "environment; in checkout mode it would be recorded as the "
                "framework version".format(distribution)
            )
    else:
        if not location or _under(location, source_dir):
            problems.append(
                "training_framework should come from the environment but is "
                "imported from {}".format(location or "nowhere")
            )
        wanted = expected_framework.get("distribution")
        if wanted is not None and distribution != wanted:
            problems.append(
                "the environment has training-framework {} but the "
                "requirements pin {}".format(distribution, wanted)
            )

    for package, error in sorted((facts.get("components") or {}).items()):
        if error is not None:
            problems.append("components package {} does not import: {}".format(package, error))
    for package in expected.get("components_packages") or []:
        if package not in (facts.get("components") or {}):
            problems.append("components package {} was not checked".format(package))

    for path, exists in sorted((facts.get("cluster_paths") or {}).items()):
        if not exists:
            problems.append("cluster data path {} does not exist".format(path))
    return problems
