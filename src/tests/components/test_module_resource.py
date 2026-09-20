"""Tests for `ModuleResource`: nn.Module resources composed of resources.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pickle

import pytest
import torch
from torch import nn

from tests.test_utils import make_config
from training_framework.components import (
    ComponentDependencyError,
    ComponentView,
    component_registry,
    ModuleResource,
    Resource,
    constructing_component,
    requires_resource,
    resource,
)
from training_framework.components.builtin.checkpointing import Checkpointer
from training_framework.engine.worker import load_session_for_worker
from training_framework.session import AnalysisSession, TrainingSession
from training_framework.session.components import SessionComponents


class _StubView(ComponentView):
    """A view over a fixed set of resources, for components under test."""

    session_type = "training"

    def __init__(self, resources=None):
        self._resources = dict(resources or {})

    def resolve_name(self, name):
        return name

    def has_resource(self, name):
        return name in self._resources

    def get_resource(self, name):
        return self._resources[name]


class PicklableEncoder(ModuleResource):
    """Declared at module scope so `pickle` can find it again."""

    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.linear(inputs)


def _declare_encoder_and_model():
    @resource("mr_encoder")
    class Encoder(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.linear = nn.Linear(4, 4)

        def forward(self, inputs):
            return self.linear(inputs)

    @requires_resource("mr_encoder")
    @resource("mr_model")
    class Model(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.mr_encoder = self.get_dependency("mr_encoder")
            self.head = nn.Linear(4, 2)

        def forward(self, inputs):
            return self.head(self.mr_encoder(inputs))

    return Encoder, Model


def _activate(config=None):
    components = SessionComponents()
    components.register_from_config(config or {"mr_model": {}, "mr_encoder": {}})
    return components


def test_a_child_is_the_same_object_as_the_registered_resource():
    _declare_encoder_and_model()
    components = _activate()

    model = components.get_resource("mr_model")
    encoder = components.get_resource("mr_encoder")

    assert model.mr_encoder is encoder
    assert model.linked_components == {"mr_encoder": "mr_encoder"}


def test_a_child_contributes_its_parameters_exactly_once():
    _declare_encoder_and_model()
    components = _activate()
    model = components.get_resource("mr_model")
    encoder = components.get_resource("mr_encoder")

    parameters = list(model.parameters())

    assert len(parameters) == 4
    for parameter in encoder.parameters():
        assert sum(parameter is other for other in parameters) == 1


def test_a_child_attached_twice_still_contributes_its_parameters_once():
    _declare_encoder_and_model()

    @requires_resource("mr_encoder")
    @resource("mr_twice")
    class Twice(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            encoder = self.get_dependency("mr_encoder")
            self.first = encoder
            self.second = encoder

    components = _activate({"mr_twice": {}, "mr_encoder": {}})
    twice = components.get_resource("mr_twice")

    assert twice.first is twice.second
    assert len(list(twice.parameters())) == 2


def test_each_component_captures_only_the_weights_it_owns():
    _declare_encoder_and_model()
    components = _activate()
    model = components.get_resource("mr_model")
    encoder = components.get_resource("mr_encoder")

    model_state = model.get_state()
    encoder_state = encoder.get_state()

    assert set(model_state["state_dict"]) == {"head.weight", "head.bias"}
    assert model_state["linked"] == {"mr_encoder": "mr_encoder"}
    assert set(encoder_state["state_dict"]) == {"linear.weight", "linear.bias"}
    assert encoder_state["linked"] == {}


def test_set_state_loads_in_place_and_keeps_parameter_identity():
    _declare_encoder_and_model()
    components = _activate()
    encoder = components.get_resource("mr_encoder")

    with torch.no_grad():
        encoder.linear.weight.fill_(0.5)
    state = encoder.get_state()
    weight = encoder.linear.weight
    with torch.no_grad():
        encoder.linear.weight.fill_(0.0)

    encoder.set_state(state)

    assert encoder.linear.weight is weight
    assert torch.equal(weight, torch.full_like(weight, 0.5))


def test_a_dependency_free_module_resource_pickles_round_trip():
    original = PicklableEncoder({})
    with torch.no_grad():
        original.linear.weight.fill_(0.25)

    restored = pickle.loads(pickle.dumps(original))

    assert torch.equal(restored.linear.weight, original.linear.weight)


def test_constructing_a_dependent_component_by_hand_is_rejected():
    _declare_encoder_and_model()

    @requires_resource("mr_encoder")
    @resource("mr_handmade")
    class Handmade(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.mr_encoder = self.get_dependency("mr_encoder")

    with pytest.raises(ComponentDependencyError, match="activate_component"):
        Handmade({})


def test_a_stub_view_is_enough_to_construct_a_component_under_test():
    _declare_encoder_and_model()

    @requires_resource("mr_encoder")
    @resource("mr_stubbed")
    class Stubbed(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.mr_encoder = self.get_dependency("mr_encoder")

    encoder = PicklableEncoder({})
    with constructing_component(_StubView({"mr_encoder": encoder})):
        stubbed = Stubbed({})

    assert stubbed.mr_encoder is encoder


def test_a_non_module_prerequisite_contributes_no_tensors():
    @resource("mr_plain")
    class Plain(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @requires_resource("mr_plain")
    @resource("mr_needs_module")
    class NeedsModule(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.plain = self.get_dependency("mr_plain")
            self.own = nn.Linear(2, 2)

    components = _activate({"mr_needs_module": {}, "mr_plain": {}})
    needs_module = components.get_resource("mr_needs_module")

    assert needs_module.plain is components.get_resource("mr_plain")
    assert needs_module.linked_components == {"mr_plain": "mr_plain"}
    assert set(needs_module.get_state()["state_dict"]) == {
        "own.weight", "own.bias",
    }


def test_a_prerequisite_may_be_held_inside_a_container_module():
    _declare_encoder_and_model()

    @requires_resource("mr_encoder")
    @resource("mr_nested")
    class Nested(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            # Not under an attribute of its own: a prerequisite is recognised
            # by identity, so it may be held anywhere in the tree.
            self.wrapper = nn.Sequential(self.get_dependency("mr_encoder"))
            self.gate = nn.Linear(4, 4)

    components = _activate({"mr_nested": {}, "mr_encoder": {}})
    nested = components.get_resource("mr_nested")

    assert nested.wrapper[0] is components.get_resource("mr_encoder")
    assert set(nested.get_state()["state_dict"]) == {"gate.weight", "gate.bias"}
    assert nested.get_state()["linked"] == {"mr_encoder": "mr_encoder"}
    # The encoder's weights are checkpointed once, by the encoder.
    components.get_state()


def test_two_components_capturing_the_same_tensor_is_rejected():
    shared = nn.Linear(4, 4)

    @resource("mr_first_owner")
    class FirstOwner(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.borrowed = shared

    @resource("mr_second_owner")
    class SecondOwner(ModuleResource):
        def __init__(self, config=None):
            # Nested, where a walk from the parent alone could miss it.
            super().__init__(config)
            self.wrapper = nn.Sequential(shared)

    components = _activate({"mr_first_owner": {}, "mr_second_owner": {}})

    with pytest.raises(
        ComponentDependencyError,
        match="are the same tensor, so it would be checkpointed twice",
    ):
        components.get_state()


def test_a_component_may_be_owned_privately_as_an_ordinary_module():
    _declare_encoder_and_model()
    encoder_class = component_registry()["mr_encoder"]

    @resource("mr_private_owner")
    class PrivateOwner(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.private = encoder_class({})
            self.stack = nn.Sequential(encoder_class({}))

    components = _activate({"mr_private_owner": {}})
    owner = components.get_resource("mr_private_owner")

    # Nobody else holds these, so the owner checkpoints them itself.
    assert set(owner.get_state()["state_dict"]) == {
        "private.linear.weight", "private.linear.bias",
        "stack.0.linear.weight", "stack.0.linear.bias",
    }
    assert owner.get_state()["linked"] == {}
    components.get_state()


def test_a_privately_owned_component_survives_a_state_round_trip(tmp_path):
    _declare_encoder_and_model()
    encoder_class = component_registry()["mr_encoder"]

    @resource("mr_private_round_trip")
    class PrivateOwner(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.private = encoder_class({})

    config = make_config(tmp_path / "private")
    config["mr_private_round_trip"] = {}
    session = TrainingSession(config)
    with torch.no_grad():
        session.get_resource("mr_private_round_trip").private.linear.weight.fill_(0.5)

    restored = TrainingSession.from_state(session.get_state())

    weight = restored.get_resource("mr_private_round_trip").private.linear.weight
    assert torch.equal(weight, torch.full((4, 4), 0.5))


def test_a_privately_owned_component_the_session_drives_is_rejected():
    @resource("mr_lifecycle")
    class WithLifecycle(ModuleResource):
        def setup(self, session):
            super().setup(session)

    @resource("mr_lifecycle_owner")
    class Owner(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.private = WithLifecycle({})

    components = _activate({"mr_lifecycle_owner": {}})
    owner = components.get_resource("mr_lifecycle_owner")

    with pytest.raises(ComponentDependencyError, match="driven by the session"):
        owner.get_state()


def test_a_privately_owned_component_with_prerequisites_is_rejected():
    _, model_class = _declare_encoder_and_model()

    # mr_model declares a prerequisite, so only the session can wire it.
    with constructing_component(_StubView({"mr_encoder": PicklableEncoder({})})):
        dependent = model_class({})

    @resource("mr_dependent_owner")
    class Owner(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.private = dependent

    with pytest.raises(ComponentDependencyError, match="driven by the session"):
        Owner({}).get_state()


def test_a_child_may_hold_components_of_its_own():
    _declare_encoder_and_model()

    @requires_resource("mr_model")
    @resource("mr_outer")
    class Outer(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.mr_model = self.get_dependency("mr_model")
            self.gate = nn.Linear(2, 2)

    components = _activate(
        {"mr_outer": {}, "mr_model": {}, "mr_encoder": {}},
    )
    outer = components.get_resource("mr_outer")

    # mr_model legitimately holds mr_encoder; only mr_model's own weights are
    # excluded here, and mr_encoder's are excluded by mr_model.
    assert set(outer.get_state()["state_dict"]) == {"gate.weight", "gate.bias"}
    assert set(components.get_resource("mr_model").get_state()["state_dict"]) == {
        "head.weight",
        "head.bias",
    }


def test_state_from_a_different_wiring_is_rejected():
    _declare_encoder_and_model()
    components = _activate()
    model = components.get_resource("mr_model")

    state = model.get_state()
    state["linked"] = {"mr_encoder": "something_else"}

    with pytest.raises(ValueError, match="was checkpointed with linked"):
        model.set_state(state)


def test_state_carrying_weights_the_component_does_not_own_is_rejected():
    _declare_encoder_and_model()
    components = _activate()
    model = components.get_resource("mr_model")
    encoder = components.get_resource("mr_encoder")

    state = model.get_state()
    state["state_dict"]["mr_encoder.linear.weight"] = encoder.linear.weight.clone()

    with pytest.raises(ValueError, match="keys it does not own"):
        model.set_state(state)


def test_state_missing_an_owned_key_is_rejected():
    _declare_encoder_and_model()
    components = _activate()
    model = components.get_resource("mr_model")

    state = model.get_state()
    del state["state_dict"]["head.bias"]

    with pytest.raises(ValueError, match="missing keys"):
        model.set_state(state)


def test_state_with_a_mismatched_shape_names_the_component():
    _declare_encoder_and_model()
    components = _activate()
    encoder = components.get_resource("mr_encoder")

    state = encoder.get_state()
    state["state_dict"]["linear.weight"] = torch.zeros(3, 3)

    with pytest.raises(ValueError, match="could not load its state"):
        encoder.set_state(state)


def _training_config(tmp_path, name):
    config = make_config(tmp_path / name)
    config["mr_model"] = {}
    config["mr_encoder"] = {}
    return config


def test_a_restored_session_rewires_the_model_without_setup(tmp_path):
    _declare_encoder_and_model()
    session = TrainingSession(_training_config(tmp_path, "round-trip"))
    with torch.no_grad():
        session.get_resource("mr_encoder").linear.weight.fill_(0.75)
        session.get_resource("mr_model").head.bias.fill_(-1.5)

    restored = TrainingSession.from_state(session.get_state())

    # No `with restored:` -- this is the `trained_model` analysis contract.
    model = restored.get_resource("mr_model")
    encoder = restored.get_resource("mr_encoder")
    assert model.mr_encoder is encoder
    assert torch.equal(encoder.linear.weight, torch.full((4, 4), 0.75))
    assert torch.equal(model.head.bias, torch.full((2,), -1.5))
    assert torch.equal(
        model(torch.ones(1, 4)),
        session.get_resource("mr_model")(torch.ones(1, 4)),
    )


def test_a_checkpoint_stores_each_tensor_once(tmp_path):
    _declare_encoder_and_model()
    session = TrainingSession(_training_config(tmp_path, "no-duplicates"))

    components_state = session.get_state()["components_state"]

    model_keys = components_state["mr_model"]["state"]["state_dict"]
    encoder_keys = components_state["mr_encoder"]["state"]["state_dict"]
    assert not any(key.startswith("mr_encoder.") for key in model_keys)
    assert set(encoder_keys) == {"linear.weight", "linear.bias"}


def test_a_child_shared_by_two_models_is_restored_as_one_instance(tmp_path):
    _declare_encoder_and_model()

    @requires_resource("mr_encoder")
    @resource("mr_second_model")
    class SecondModel(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.mr_encoder = self.get_dependency("mr_encoder")
            self.tail = nn.Linear(4, 1)

    config = _training_config(tmp_path, "shared-child")
    config["mr_second_model"] = {}
    session = TrainingSession(config)

    restored = TrainingSession.from_state(session.get_state())

    encoder = restored.get_resource("mr_encoder")
    assert restored.get_resource("mr_model").mr_encoder is encoder
    assert restored.get_resource("mr_second_model").mr_encoder is encoder


def test_pickling_uses_the_stateful_reconstruction_envelope():
    # nn.Module defines __getstate__/__setstate__ and would otherwise shadow
    # Stateful's envelope through the MRO.
    encoder = PicklableEncoder({})

    state = encoder.__getstate__()

    assert state["__training_framework_pickle_version__"] == 1
    assert state["init_args"] == {"args": ({},), "kwargs": {}}
    assert set(state["state"]["state_dict"]) == {"linear.weight", "linear.bias"}


def test_a_saved_checkpoint_restores_the_composed_model(tmp_path):
    _declare_encoder_and_model()
    session = TrainingSession(_training_config(tmp_path, "checkpoint"))
    with session:
        model = session.get_resource("mr_model")
        with torch.no_grad():
            session.get_resource("mr_encoder").linear.bias.fill_(0.125)
        expected = model.eval()(torch.ones(1, 4))

    checkpoint_path = tmp_path / "linked.pt"
    torch.save(session, checkpoint_path)
    restored_session = Checkpointer.load_checkpoint(checkpoint_path)

    restored_model = restored_session.get_resource("mr_model")
    assert restored_model.mr_encoder is restored_session.get_resource("mr_encoder")
    torch.testing.assert_close(restored_model.eval()(torch.ones(1, 4)), expected)


def _ddp_config(tmp_path, name, *, world_size=1):
    config = make_config(tmp_path / name)
    config["mr_encoder"] = {}
    config["model"] = {}
    config["ddp"] = {
        "world_size": world_size,
        "backend": "gloo",
        "master_addr": "127.0.0.1",
        "master_port": "29500",
        "parallel_components": ["model"],
    }
    return config


def _declare_named_model():
    @resource("mr_encoder")
    class Encoder(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.linear = nn.Linear(4, 4)

        def forward(self, inputs):
            return self.linear(inputs)

    @requires_resource("mr_encoder")
    @resource("model", session_type="training")
    @resource("model", session_type="analysis")
    class Model(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.mr_encoder = self.get_dependency("mr_encoder")
            self.head = nn.Linear(4, 2)

        def forward(self, inputs):
            return self.head(self.mr_encoder(inputs))

    return Encoder, Model


@pytest.mark.parametrize("rank", [0, 1])
def test_a_worker_builds_the_rank_specific_ddp_resource(tmp_path, rank):
    _declare_named_model()
    session = TrainingSession(_ddp_config(tmp_path, f"worker-{rank}", world_size=2))

    worker_session = load_session_for_worker(
        pickle.loads(pickle.dumps(session.get_state())),
        rank,
    )

    assert worker_session.get_resource("ddp").rank == rank
    model = worker_session.get_resource("model")
    assert model.mr_encoder is worker_session.get_resource("mr_encoder")
    assert len(list(model.parameters())) == 4


def test_a_composed_model_is_loaded_for_analysis_by_trained_model(tmp_path):
    _declare_named_model()
    config = make_config(tmp_path / "analysis-source")
    config["mr_encoder"] = {}
    config["model"] = {}
    session = TrainingSession(config)
    with session:
        trained = session.get_resource("model")
        with torch.no_grad():
            session.get_resource("mr_encoder").linear.weight.fill_(0.3)
        expected = trained.eval()(torch.ones(2, 4))

    checkpoint_path = tmp_path / "training.pt"
    torch.save(session, checkpoint_path)

    analysis = AnalysisSession({
        "session_config": {
            "rng_seed": 5,
            "sessions_dir": str(tmp_path / "analysis"),
            "max_iterations": 1,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
        },
        "trained_model": {"model_checkpoint_path": str(checkpoint_path)},
    })
    with analysis:
        model = analysis.get_resource("trained_model").model
        assert model.mr_encoder.linear.weight.allclose(torch.full((4, 4), 0.3))
        torch.testing.assert_close(model(torch.ones(2, 4)), expected)


def test_setup_moves_the_whole_attached_tree_to_the_session_device(tmp_path):
    _declare_encoder_and_model()
    session = TrainingSession(_training_config(tmp_path, "device"))
    model = session.get_resource("mr_model")
    encoder_weight = session.get_resource("mr_encoder").linear.weight

    with session:
        assert model.head.weight.device == session.device
        assert encoder_weight.device == session.device
        # The optimizer's view of the child survives the move.
        assert session.get_resource("mr_encoder").linear.weight is encoder_weight
