import ast
import inspect

import pytest

from training_framework.hpc.remote import preflight

SOURCE = "/b/source"
IDENTITY = {"implementation": "cpython", "version": "3.12.4",
            "executable": "/opt/bin/python3", "platform": "linux-x86_64"}


def facts(**changes):
    data = {
        "protocol_versions": [1],
        "interpreter": dict(IDENTITY),
        "framework": {"file": "/env/lib/site-packages/training_framework/__init__.py",
                      "distribution": "0.5.0"},
        "components": {"my_project.components": None},
        "cluster_paths": {"/data/x": True},
    }
    data.update(changes)
    return data


def expected(**changes):
    data = {
        "protocol_version": 1,
        "interpreter": {"source": "site", "identity": dict(IDENTITY), "version_prefix": None},
        "framework": {"location": "environment", "distribution": "0.5.0"},
        "components_packages": ["my_project.components"],
        "cluster_paths": ["/data/x"],
    }
    data.update(changes)
    return data


def test_matching_facts_pass():
    assert preflight.compare(facts(), expected(), SOURCE) == []


def test_an_unsupported_protocol_names_both_versions():
    problems = preflight.compare(facts(protocol_versions=[2, 3]), expected(), SOURCE)
    assert "[2, 3]" in problems[0] and "uses 1" in problems[0]


@pytest.mark.parametrize("field", ["implementation", "version", "executable", "platform"])
def test_every_identity_field_is_compared(field):
    changed = dict(IDENTITY, **{field: "different"})
    problems = preflight.compare(facts(interpreter=changed), expected(), SOURCE)
    assert len(problems) == 1 and field in problems[0]


def test_a_pinned_version_prefix_is_compared():
    pinned = expected(interpreter={"source": "uv", "identity": None, "version_prefix": "3.13"})
    assert preflight.compare(facts(interpreter=dict(IDENTITY, version="3.13.1")), pinned, SOURCE) == []
    assert preflight.compare(facts(interpreter=dict(IDENTITY, version="3.130.0")), pinned, SOURCE)


def test_checkout_mode_needs_the_checkout_and_no_distribution():
    checkout = expected(framework={"location": "checkout", "distribution": None})
    good = facts(framework={"file": f"{SOURCE}/src/training_framework/__init__.py", "distribution": None})
    assert preflight.compare(good, checkout, SOURCE) == []
    problems = preflight.compare(facts(), checkout, SOURCE)
    assert any("instead of the checkout" in p for p in problems)
    assert any("distribution (0.5.0) is installed" in p for p in problems)


def test_dependency_mode_needs_the_environment_and_the_pinned_version():
    from_checkout = facts(framework={"file": f"{SOURCE}/training_framework/__init__.py", "distribution": "0.5.0"})
    assert any("should come from the environment" in p
               for p in preflight.compare(from_checkout, expected(), SOURCE))
    wrong = facts(framework={"file": "/env/training_framework/__init__.py", "distribution": "0.4.0"})
    assert any("pin 0.5.0" in p for p in preflight.compare(wrong, expected(), SOURCE))


def test_components_and_cluster_paths_are_reported():
    problems = preflight.compare(
        facts(components={"my_project.components": "ImportError: nope"},
              cluster_paths={"/data/x": False}),
        expected(), SOURCE,
    )
    assert any("does not import" in p for p in problems)
    assert any("/data/x does not exist" in p for p in problems)


def test_the_module_uses_only_python_3_8_syntax_and_the_standard_library():
    source = inspect.getsource(preflight)
    tree = ast.parse(source, feature_version=(3, 8))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])
    }
    assert imported <= {"typing"}
