"""Every self-supervised built-in survives a pickle round trip and a
checkpoint: a trained session's components come back with the same weights
or state, and behave the same."""

from __future__ import annotations

import pickle

import pytest
import torch

from tests.integration.integration_ssl_components import byol_config, dino_config
from tests.test_utils import component_named
from training_framework.components.builtin import Checkpointer

from .helpers import quiet_session, run

MODULES = [
    "projection_head#student",
    "backbone_with_head#student",
    "multi_view#student",
    "ema_teacher",
]
STEPS = ["ema_update", "dino_distillation_loss"]


@pytest.fixture
def trained(tmp_path):
    session = quiet_session(dino_config(tmp_path / "run", max_iterations=3))
    run(session, 2)
    return session


def _same_tensors(left: dict, right: dict) -> None:
    assert list(left) == list(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], msg=key)


def _same_state(left, right) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], msg=key)
        else:
            assert left[key] == right[key], key


VIEWS = [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)]


@pytest.mark.parametrize("name", MODULES)
def test_a_pickled_module_has_the_same_weights_and_output(trained, name):
    module = component_named(trained, name).eval()
    copy = pickle.loads(pickle.dumps(module)).eval()

    _same_tensors(module.state_dict(), copy.state_dict())
    inputs = (
        torch.randn(2, 16) if name.startswith("projection_head")
        else VIEWS[0] if name.startswith("backbone_with_head")
        else VIEWS
    )
    torch.testing.assert_close(copy(inputs), module(inputs))


@pytest.mark.parametrize("name", STEPS)
def test_a_pickled_step_has_the_same_state(trained, name):
    step = component_named(trained, name)

    copy = pickle.loads(pickle.dumps(step))

    _same_state(copy.get_state(), step.get_state())


def test_a_checkpointed_session_restores_every_component(trained, tmp_path):
    path = Checkpointer.save_checkpoint(trained, tmp_path / "checkpoint")

    restored = Checkpointer.load_checkpoint(path, restore_rng=False)

    for name in MODULES:
        _same_tensors(
            component_named(trained, name).state_dict(),
            component_named(restored, name).state_dict(),
        )
    for name in STEPS:
        _same_state(
            component_named(trained, name).get_state(),
            component_named(restored, name).get_state(),
        )


def test_a_checkpointed_byol_model_restores_its_nested_networks(tmp_path):
    session = quiet_session(byol_config(tmp_path / "run", max_iterations=3))
    run(session, 2)
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")

    restored = Checkpointer.load_checkpoint(path, restore_rng=False)

    _same_tensors(
        component_named(session, "ema_teacher").state_dict(),
        component_named(restored, "ema_teacher").state_dict(),
    )
