"""The ddp resource leaves the process environment as it found it.

`MASTER_ADDR` / `MASTER_PORT` are process-wide, and launch topology lets the
environment outrank a checkpoint's stored port on purpose. A value written by
one session therefore became the next session's rendezvous in the same
process. The resource hands the address to `init_process_group` directly
instead.

Torch's process-group calls are replaced: a hand-built session has no rank
of its own, and the spawned integration tests cover a real rendezvous.
"""

import os

import pytest
import torch
from torch import nn

from tests.test_utils import make_config
from training_framework.components import Resource, resource
from training_framework.components.builtin import distributed
from training_framework.session import TrainingSession

_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT")


@pytest.fixture
def process_group(monkeypatch):
    """Record each init_process_group call and the environment it saw."""
    calls = {"init": [], "destroyed": 0}

    def init_process_group(**kwargs):
        calls["init"].append({
            "kwargs": kwargs,
            "env": _environment(),
        })

    def destroy_process_group():
        calls["destroyed"] += 1

    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        torch.distributed, "init_process_group", init_process_group,
    )
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", destroy_process_group,
    )
    return calls


def _environment():
    return {key: os.environ.get(key) for key in _ENV_KEYS}


def _session(tmp_path, *, master_addr="127.0.0.1", model_is_module=True):
    if model_is_module:
        @resource("env_model")
        class Model(nn.Linear, Resource):
            def __init__(self, config=None):
                nn.Linear.__init__(self, 1, 1)

            def setup(self, session) -> None:
                pass

            def teardown(self, session) -> None:
                pass
    else:
        @resource("env_model")
        class Model(Resource):
            """Not an nn.Module, so wrapping it in DDP fails after
            init_process_group has already run."""

            def setup(self, session) -> None:
                pass

            def teardown(self, session) -> None:
                pass

    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"model": "env_model"}
    config["env_model"] = {}
    config["ddp"] = {
        "world_size": 1,
        "backend": "gloo",
        "master_addr": master_addr,
        "master_port": "29517",
    }
    return TrainingSession(config)


class _FakeDDP:
    def __init__(self, module, device_ids=None):
        if not isinstance(module, nn.Module):
            raise TypeError("DDP needs an nn.Module")
        self.module = module


def test_a_session_leaves_the_environment_untouched(
        tmp_path, process_group, monkeypatch,
):
    monkeypatch.setattr(distributed, "DDP", _FakeDDP)
    session = _session(tmp_path)

    with session:
        pass

    [call] = process_group["init"]
    assert call["kwargs"]["init_method"] == "tcp://127.0.0.1:29517"
    assert call["env"] == {"MASTER_ADDR": None, "MASTER_PORT": None}
    assert process_group["destroyed"] == 1
    assert _environment() == {"MASTER_ADDR": None, "MASTER_PORT": None}


def test_a_setup_that_fails_after_joining_leaves_the_environment_untouched(
        tmp_path, process_group, monkeypatch,
):
    monkeypatch.setattr(distributed, "DDP", _FakeDDP)
    session = _session(tmp_path, model_is_module=False)

    with pytest.raises(TypeError, match="nn.Module"):
        with session:
            pass

    assert len(process_group["init"]) == 1
    assert process_group["destroyed"] == 1
    assert _environment() == {"MASTER_ADDR": None, "MASTER_PORT": None}


def test_an_existing_environment_is_neither_read_nor_overwritten(
        tmp_path, process_group, monkeypatch,
):
    monkeypatch.setattr(distributed, "DDP", _FakeDDP)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.9")
    monkeypatch.setenv("MASTER_PORT", "40000")
    session = _session(tmp_path)

    with session:
        pass

    [call] = process_group["init"]
    assert call["kwargs"]["init_method"] == "tcp://127.0.0.1:29517"
    assert _environment() == {"MASTER_ADDR": "10.0.0.9", "MASTER_PORT": "40000"}


def test_an_ipv6_master_address_is_bracketed(
        tmp_path, process_group, monkeypatch,
):
    monkeypatch.setattr(distributed, "DDP", _FakeDDP)
    session = _session(tmp_path, master_addr="::1")

    with session:
        pass

    [call] = process_group["init"]
    assert call["kwargs"]["init_method"] == "tcp://[::1]:29517"
