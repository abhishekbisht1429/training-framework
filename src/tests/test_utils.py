"""Shared helpers for the commit-3d45 replacement tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


COMPONENTS_PACKAGE = "tests.test_components"


def register_test_components() -> ModuleType:
    """Register test-only components after the fixture restores the built-ins.

    The test component module is imported normally on the first call and
    reloaded on later calls. This is also compatible with a fresh spawned
    interpreter, where the module is imported by TrainingSession itself.
    """

    existing = sys.modules.get(COMPONENTS_PACKAGE)
    if existing is None:
        return importlib.import_module(COMPONENTS_PACKAGE)
    return importlib.reload(existing)


def session_config(
    root: Path,
    *,
    max_iterations: int,
    event_path: Path | None = None,
    seed: int = 923,
    include_metrics: bool = False,
) -> dict[str, Any]:
    model_config: dict[str, Any] = {
        "initial_weight": 0.25,
        "learning_rate": 0.08,
        "momentum": 0.9,
    }
    train_config: dict[str, Any] = {"noise_scale": 0.07}
    if event_path is not None:
        train_config["event_path"] = str(event_path)

    config: dict[str, Any] = {
        "session_config": {
            "rng_seed": seed,
            "sessions_dir": str(root),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "it_3d45_model": model_config,
        "it_3d45_train": train_config,
    }
    if include_metrics:
        metrics_config: dict[str, Any] = {"call_every": 1}
        if event_path is not None:
            model_config["event_path"] = str(event_path)
            metrics_config["event_path"] = str(event_path)
        config["it_3d45_metrics"] = metrics_config
    return config


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def iteration_events(path: Path) -> list[dict[str, Any]]:
    return [event for event in read_events(path) if event["event"] == "iteration"]


def make_config(tmp_path, max_iterations=2, seed=123):
    return {
        "session_config": {
            "rng_seed": seed,
            "sessions_dir": str(tmp_path),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
        }
    }


def build_session(
        tmp_path,
        components=None,
        *,
        session_type="training",
        bindings=None,
):
    """Build a session holding `components`, the way configuration does.

    `components` maps top-level config keys to their mappings. An analysis
    session needs a `trained_model` mapping to construct at all. Unless the
    caller configures or binds one, it gets an empty placeholder file:
    construction only checks that the file exists, and the checkpoint is not
    read until `setup`.
    """
    from training_framework.session import AnalysisSession, TrainingSession

    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    if bindings is not None:
        config["component_bindings"] = bindings
    config.update(components or {})
    if session_type == "training":
        return TrainingSession(config)
    if "trained_model" not in config and "trained_model" not in (bindings or {}):
        placeholder = tmp_path / "placeholder-checkpoint.pt"
        placeholder.touch()
        config["trained_model"] = {"model_checkpoint_path": str(placeholder)}
    return AnalysisSession(config)


def configurator_for(tmp_path, monkeypatch, *sessions):
    """Return a `Configurator` reading `sessions` from a config file, the way
    the command line hands it one."""
    import yaml

    from training_framework.engine import Configurator

    path = tmp_path / "configurator.yaml"
    path.write_text(yaml.safe_dump({"sessions": list(sessions)}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
    return Configurator()


def inject_dependencies(component, **dependencies):
    """Give a hand-built component the prerequisites a session would inject.

    Mirrors what `SessionComponents` does when a component built outside it
    is registered: the prerequisites go into the instance `__dict__`, where
    `Component.get_dependency` finds them. Lets a test drive one lifecycle
    method with a fake session and stub prerequisites.
    """
    from training_framework.components import Component

    component.__dict__[Component.DEPENDENCIES_ATTR] = dict(dependencies)
    return component


def _named(components, session, name, kind):
    target = session.resolve_component_name(name)
    for component in components:
        if component.name == target:
            return component
    raise KeyError(f"{name} not found in {kind}!")


def all_components(session):
    """Every resource, hook and step the session holds."""
    return (
        session.get_all_resources()
        + session.get_all_hooks()
        + session.get_all_steps()
    )


def component_names(session) -> set[str]:
    """The instance names of everything the session holds."""
    return {component.name for component in all_components(session)}


def resource_named(session, name):
    """Return the resource `name` refers to, for a test to inspect.

    Code outside a component has no consumer to resolve for, so this applies
    the session-wide bindings -- a role such as `model` finds the
    implementation bound to it -- and matches the instance name exactly.
    Components take their prerequisites with `get_dependency` instead.
    """
    return _named(session.get_all_resources(), session, name, "resources")


def component_named(session, name):
    """Like `resource_named`, but also finds hooks and steps."""
    return _named(all_components(session), session, name, "components")


def has_resource_named(session, name) -> bool:
    """Whether `resource_named(session, name)` would find a resource."""
    try:
        resource_named(session, name)
    except KeyError:
        return False
    return True
