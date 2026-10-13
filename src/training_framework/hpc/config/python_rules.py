"""Python version and interpreter identity rules (plan section 9.4).

Every use of `packaging` happens here, on the workstation. The remote side
only reports facts that are compared exactly with the expectations computed
below.
"""

from __future__ import annotations

from dataclasses import dataclass

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from training_framework.hpc.config.schema import EnvSettings, PythonIdentity


@dataclass(frozen=True)
class InterpreterExpectation:
    """What preflight item 1 compares against. `identity` for a declared
    `site` interpreter; `version_prefix` for `uv`/`conda` (the pin); neither
    for `env.existing`, which is probed at submit."""

    source: str
    identity: PythonIdentity | None
    version_prefix: str | None

    def as_json(self) -> dict:
        return {
            "source": self.source,
            "identity": self.identity.as_json() if self.identity else None,
            "version_prefix": self.version_prefix,
        }

    def key_part(self) -> dict | None:
        """Interpreter identity as part of the environment key."""
        if self.identity is not None:
            return {"site": self.identity.as_json()}
        if self.version_prefix is not None:
            return {self.source: self.version_prefix}
        return None


def parse_requires_python(text: str | None) -> tuple[SpecifierSet | None, str | None]:
    if text is None:
        return None, None
    try:
        return SpecifierSet(text), None
    except InvalidSpecifier as error:
        return None, f"requires-python '{text}' is not a valid specifier ({error})"


def _decidable(specifiers: SpecifierSet, pin: str) -> bool:
    """Whether a minor pin X.Y decides `specifiers` for every X.Y.z."""
    major_minor = tuple(int(part) for part in pin.split("."))
    for spec in specifiers:
        version = spec.version.rstrip(".*")
        try:
            release = Version(version).release
        except InvalidVersion:
            return False
        if len(release) >= 3 and release[:2] == major_minor and not spec.version.endswith(".*"):
            return False
    return True


def interpreter_expectation(
        env: EnvSettings,
        pin: str | None,
        requires_python: SpecifierSet | None,
        managed: bool,
) -> tuple[InterpreterExpectation, list[str]]:
    """The expected interpreter and the problems with the configuration."""
    problems: list[str] = []
    if not managed:
        return InterpreterExpectation("existing", None, None), problems

    if env.backend == "conda" and env.python_source == "site":
        # `conda create python=<pin>` installs the environment's own Python:
        # it is never the site interpreter preflight would compare it with,
        # and 'site' does not require the pin conda needs.
        problems.append(
            "profile env.backend 'conda' installs its own Python "
            "(conda create python=<environment.python>), so it cannot use "
            "python_source 'site'; set env.python_source to 'conda' (or "
            "'uv') and environment.python in the manifest, or use "
            "backend 'venv' with the site interpreter"
        )
        return InterpreterExpectation("site", None, None), problems

    if env.python_source == "site":
        identity = env.site_python_identity
        if identity is None:
            problems.append(
                "profile env.site_python_identity is required for a managed "
                "environment with python_source 'site' (run "
                "`python -m training_framework.hpc probe-python` and paste "
                "its output)"
            )
            return InterpreterExpectation("site", None, None), problems
        if pin is not None and not _matches_pin(identity.version, pin):
            problems.append(
                f"environment.python pins {pin}, but the profile's "
                f"site_python_identity is Python {identity.version}"
            )
        if requires_python is not None and not requires_python.contains(
                identity.version, prereleases=True,
        ):
            problems.append(
                f"the site interpreter (Python {identity.version}) does not "
                f"satisfy requires-python '{requires_python}'"
            )
        return InterpreterExpectation("site", identity, None), problems

    if pin is None:
        problems.append(
            f"environment.python is required with python_source "
            f"'{env.python_source}' (the interpreter it installs must be "
            "known offline)"
        )
        return InterpreterExpectation(env.python_source, None, None), problems
    if requires_python is not None:
        if pin.count(".") == 1 and not _decidable(requires_python, pin):
            problems.append(
                f"environment.python '{pin}' cannot decide requires-python "
                f"'{requires_python}' (it has a patch-level bound); pin a "
                "full version X.Y.Z"
            )
        else:
            probe = pin if pin.count(".") == 2 else f"{pin}.0"
            if not requires_python.contains(probe, prereleases=True):
                problems.append(
                    f"environment.python '{pin}' does not satisfy "
                    f"requires-python '{requires_python}'"
                )
    return InterpreterExpectation(env.python_source, None, pin), problems


def _matches_pin(version: str, pin: str) -> bool:
    return version == pin or version.startswith(pin + ".")
