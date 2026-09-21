"""Reading one component out of a checkpoint.

A component that needs something another run produced -- the analysis
session's `trained_model` is the built-in case -- gets it through
`Checkpointer.load_component`, which resolves the name through the
checkpoint's own bindings. This replaces reaching into the loaded session's
private component container.
"""

import pytest
import torch
from torch import nn

from tests.test_utils import make_config
from training_framework.components import ComponentDependencyError, Resource, resource
from training_framework.components.builtin.checkpointing import Checkpointer
from training_framework.session import TrainingSession


class LoadableModel(nn.Linear, Resource):
    """Declared at module scope so a checkpoint holding it can be unpickled."""

    def __init__(self, config=None):
        nn.Linear.__init__(self, 1, 1)
        self._config = config or {}
        with torch.no_grad():
            self.weight.fill_(float(self._config.get("weight", 0.0)))

    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass


def save_checkpoint(tmp_path, **instances):
    resource("loadable_model")(LoadableModel)
    config = make_config(tmp_path / "source")
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"model": "loadable_model"}
    config.update(instances)
    path = tmp_path / "checkpoint.pt"
    torch.save(TrainingSession(config), path)
    return path


def test_a_role_resolves_through_the_checkpoints_own_bindings(tmp_path):
    path = save_checkpoint(tmp_path, loadable_model={"weight": 3.0})

    model = Checkpointer.load_component(path, "model")

    assert isinstance(model, LoadableModel)
    assert model.weight.item() == 3.0


def test_the_checkpoint_must_hold_the_session_type_asked_for(tmp_path):
    path = save_checkpoint(tmp_path, loadable_model={})

    assert Checkpointer.load_component(path, "model", session_type="training")
    with pytest.raises(ValueError, match="must contain an analysis session"):
        Checkpointer.load_component(path, "model", session_type="analysis")


def test_a_missing_component_is_a_key_error(tmp_path):
    path = save_checkpoint(tmp_path, loadable_model={})

    with pytest.raises(KeyError):
        Checkpointer.load_component(path, "no_such_component")


def test_two_candidates_are_reported_not_chosen_between(tmp_path):
    path = save_checkpoint(
        tmp_path,
        **{"loadable_model#a": {}, "loadable_model#b": {}},
    )

    with pytest.raises(ComponentDependencyError):
        Checkpointer.load_component(path, "model")


def _draw_after(seed, load):
    torch.manual_seed(seed)
    load()
    return torch.rand(1).item()


def test_loading_a_component_does_not_adopt_the_checkpoints_rng(tmp_path):
    """Rebuilding the checkpoint's components draws from the generator either
    way; what matters is whether the caller's seed still decides what comes
    next, or the checkpoint's saved generator state does."""
    path = save_checkpoint(tmp_path, loadable_model={})

    # Loading the whole session adopts its RNG: the caller's seed is lost.
    adopted = [
        _draw_after(seed, lambda: Checkpointer.load_checkpoint(path))
        for seed in (1, 2)
    ]
    assert adopted[0] == adopted[1]

    followed = [
        _draw_after(seed, lambda: Checkpointer.load_component(path, "model"))
        for seed in (1, 2)
    ]
    assert followed[0] != followed[1]
