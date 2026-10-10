import pytest

from training_framework.hpc.config.python_rules import (
    interpreter_expectation,
    parse_requires_python,
)
from training_framework.hpc.config.schema import EnvSettings, PythonIdentity

IDENTITY = PythonIdentity("cpython", "3.12.4", "/opt/bin/python3", "linux-x86_64")


def expect(env, pin=None, spec=">=3.12", managed=True):
    requires, problem = parse_requires_python(spec)
    assert problem is None
    return interpreter_expectation(env, pin, requires, managed)


def test_site_uses_the_declared_identity():
    expectation, problems = expect(EnvSettings(site_python_identity=IDENTITY))
    assert not problems
    assert expectation.identity == IDENTITY
    assert expectation.key_part() == {"site": IDENTITY.as_json()}


def test_different_declared_identities_give_different_key_parts():
    other = PythonIdentity("cpython", "3.12.5", "/opt/bin/python3", "linux-x86_64")
    first, _ = expect(EnvSettings(site_python_identity=IDENTITY))
    second, _ = expect(EnvSettings(site_python_identity=other))
    assert first.key_part() != second.key_part()


def test_site_without_a_declared_identity_is_refused():
    _, problems = expect(EnvSettings())
    assert "site_python_identity is required" in problems[0]
    assert "probe-python" in problems[0]


def test_site_identity_must_satisfy_requires_python_and_the_pin():
    _, problems = expect(EnvSettings(site_python_identity=IDENTITY), pin="3.13", spec=">=3.13")
    assert any("pins 3.13" in problem for problem in problems)
    assert any("does not satisfy" in problem for problem in problems)


@pytest.mark.parametrize("source", ["uv", "conda"])
def test_uv_and_conda_need_a_pin_and_derive_identity_from_it(source):
    _, problems = expect(EnvSettings(python_source=source))
    assert "environment.python is required" in problems[0]
    expectation, problems = expect(EnvSettings(python_source=source), pin="3.13")
    assert not problems
    assert expectation.version_prefix == "3.13"
    assert expectation.key_part() == {source: "3.13"}


def test_a_minor_pin_that_cannot_decide_a_patch_bound_is_refused():
    _, problems = expect(EnvSettings(python_source="uv"), pin="3.12", spec=">=3.12.2")
    assert "cannot decide" in problems[0]
    _, problems = expect(EnvSettings(python_source="uv"), pin="3.12.5", spec=">=3.12.2")
    assert not problems
    _, problems = expect(EnvSettings(python_source="uv"), pin="3.11", spec=">=3.12")
    assert "does not satisfy" in problems[0]


def test_an_existing_environment_is_probed_at_submit_not_keyed():
    expectation, problems = expect(EnvSettings(existing="/env"), managed=False)
    assert not problems
    assert expectation.identity is None and expectation.key_part() is None


def test_an_invalid_requires_python_is_reported():
    _, problem = parse_requires_python("python3 please")
    assert "not a valid specifier" in problem
