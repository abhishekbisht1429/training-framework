from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

from training_framework.components import (
    Step,
    component_registry,
    requires_resource,
    role_registry,
    step,
    topological_sort_of_components,
)
from training_framework.components.builtin import TrainedModel
from training_framework.components.builtin.transformer import (
    AttentionPooling,
    AttentionPoolingFactory,
    ClassToken,
    ConditionedPoolingQueryFactory,
    ConditionedQuery,
    ConvPatchEmbeddingFactory,
    LearnedPoolingQueryFactory,
    LearnedPositionalEmbedding2D,
    LearnedPositionalEmbedding2DFactory,
    LearnedQuery,
    ModuleFactory,
    PatchEmbedding,
    PatchTransformer,
    PooledPatchTransformer,
    SinusoidalPositionalEmbedding2D,
    SinusoidalPositionalEmbedding2DFactory,
    TokenReduction,
    TorchTransformerEncoderFactory,
    TransformerEncoder,
)
from training_framework.session import AnalysisSession, TrainingSession


EMBED_DIM = 8

FACTORY_CONFIGS = {
    "conv_patch_embedding": {"in_channels": 3, "patch_size": 4, "embed_dim": EMBED_DIM},
    "learned_positional_embedding_2d": {"grid_size": [2, 2], "embed_dim": EMBED_DIM},
    "sinusoidal_positional_embedding_2d": {"embed_dim": EMBED_DIM},
    "torch_transformer_encoder": {
        "embed_dim": EMBED_DIM,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
    },
    "attention_pooling": {"embed_dim": EMBED_DIM, "num_heads": 2},
    "learned_pooling_query": {"embed_dim": EMBED_DIM, "num_queries": 2},
    "conditioned_pooling_query": {
        "embed_dim": EMBED_DIM,
        "inputs": {
            "obj_patch": {
                "module": "training_framework.components.builtin.transformer.PatchEmbedding",
                "in_channels": 3,
                "patch_size": 4,
                "embed_dim": EMBED_DIM,
                "reduce": "mean",
            },
            "obj_patch_location": {
                "module": "torch.nn.Linear",
                "in_features": 2,
                "out_features": EMBED_DIM,
            },
        },
        "hidden_dims": [EMBED_DIM],
    },
}


def _session_config(root, *, max_iterations=1):
    return {
        "rng_seed": 3,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def _pooled_config(tmp_path, *, positional="learned_positional_embedding_2d",
                   query="conditioned_pooling_query", model="pooled_patch_transformer",
                   max_iterations=1, extra=None):
    bindings = {
        "model": model,
        "patch_embedding": "conv_patch_embedding",
        "positional_embedding": positional,
        "sequence_encoder": "torch_transformer_encoder",
    }
    factories = ["conv_patch_embedding", positional, "torch_transformer_encoder"]
    if model == "pooled_patch_transformer":
        bindings.update({"pooling": "attention_pooling", "pooling_query": query})
        factories += ["attention_pooling", query]
    config = {
        "session_config": _session_config(tmp_path / "training", max_iterations=max_iterations),
        "component_bindings": bindings,
        model: {},
        **{name: FACTORY_CONFIGS[name] for name in factories},
    }
    config.update(extra or {})
    return config


def _conditioning(batch_size=2):
    return {
        "obj_patch": torch.randn(batch_size, 3, 4, 4),
        "obj_patch_location": torch.rand(batch_size, 2),
    }


class _FakeSession:
    device = torch.device("cpu")

    def __init__(self, resources):
        self._resources = resources

    def get_resource(self, name):
        return self._resources[name]


def _fake_session_for(model_class, **overrides):
    resources = {
        "patch_embedding": ConvPatchEmbeddingFactory(FACTORY_CONFIGS["conv_patch_embedding"]),
        "positional_embedding": LearnedPositionalEmbedding2DFactory(
            FACTORY_CONFIGS["learned_positional_embedding_2d"]
        ),
        "sequence_encoder": TorchTransformerEncoderFactory(
            FACTORY_CONFIGS["torch_transformer_encoder"]
        ),
    }
    if model_class is PooledPatchTransformer:
        resources["pooling"] = AttentionPoolingFactory(FACTORY_CONFIGS["attention_pooling"])
        resources["pooling_query"] = LearnedPoolingQueryFactory(
            FACTORY_CONFIGS["learned_pooling_query"]
        )
    resources.update(overrides)
    return _FakeSession(resources)


# -- registration -------------------------------------------------------


def test_transformer_components_and_roles_are_registered():
    registry = component_registry()
    for name, factory_class in {
        "conv_patch_embedding": ConvPatchEmbeddingFactory,
        "learned_positional_embedding_2d": LearnedPositionalEmbedding2DFactory,
        "sinusoidal_positional_embedding_2d": SinusoidalPositionalEmbedding2DFactory,
        "torch_transformer_encoder": TorchTransformerEncoderFactory,
        "attention_pooling": AttentionPoolingFactory,
        "learned_pooling_query": LearnedPoolingQueryFactory,
        "conditioned_pooling_query": ConditionedPoolingQueryFactory,
    }.items():
        assert registry[name] is factory_class
    for session_type in ("training", "analysis"):
        scoped = component_registry(session_type)
        assert scoped["patch_transformer"] is PatchTransformer
        assert scoped["pooled_patch_transformer"] is PooledPatchTransformer
    # Composites need roles without defaults, so they stay out of the shared
    # scope, which must remain sortable on its own.
    assert "patch_transformer" not in registry
    topological_sort_of_components()
    for role_name in (
        "patch_embedding",
        "positional_embedding",
        "sequence_encoder",
        "pooling",
        "pooling_query",
    ):
        assert role_name in role_registry()
    assert set(PatchTransformer.required_resources) == {
        "patch_embedding", "positional_embedding", "sequence_encoder",
    }
    assert set(PooledPatchTransformer.required_resources) == set(PooledPatchTransformer.block_roles)


# -- factories ----------------------------------------------------------


@pytest.mark.parametrize(
    "factory_class, config_name, module_class",
    [
        (ConvPatchEmbeddingFactory, "conv_patch_embedding", PatchEmbedding),
        (LearnedPositionalEmbedding2DFactory, "learned_positional_embedding_2d",
         LearnedPositionalEmbedding2D),
        (SinusoidalPositionalEmbedding2DFactory, "sinusoidal_positional_embedding_2d",
         SinusoidalPositionalEmbedding2D),
        (TorchTransformerEncoderFactory, "torch_transformer_encoder", TransformerEncoder),
        (AttentionPoolingFactory, "attention_pooling", AttentionPooling),
        (LearnedPoolingQueryFactory, "learned_pooling_query", LearnedQuery),
        (ConditionedPoolingQueryFactory, "conditioned_pooling_query", ConditionedQuery),
    ],
)
def test_factory_builds_new_real_modules(factory_class, config_name, module_class):
    factory = factory_class(FACTORY_CONFIGS[config_name])

    first, second = factory.build(), factory.build()

    assert type(first) is module_class
    assert first is not second
    assert all(not p.is_meta for p in first.parameters())
    assert not isinstance(factory, nn.Module)  # the factory owns no weights


def test_factory_reports_invalid_config_at_construction():
    with pytest.raises(TypeError, match="config must be a mapping"):
        ConvPatchEmbeddingFactory([1, 2])
    with pytest.raises(TypeError, match="Invalid conv_patch_embedding config: .*'embed_dim'"):
        ConvPatchEmbeddingFactory({"in_channels": 3, "patch_size": 4})
    with pytest.raises(TypeError, match="unexpected keyword argument 'depth'"):
        TorchTransformerEncoderFactory({**FACTORY_CONFIGS["torch_transformer_encoder"], "depth": 2})
    with pytest.raises(ValueError, match="Invalid torch_transformer_encoder config: .*num_heads"):
        TorchTransformerEncoderFactory({**FACTORY_CONFIGS["torch_transformer_encoder"], "num_heads": 3})


def test_factory_config_is_isolated_from_callers():
    config = {
        "embed_dim": EMBED_DIM,
        "inputs": {
            "x": {"module": "torch.nn.Linear", "in_features": 2, "out_features": EMBED_DIM},
        },
    }
    factory = ConditionedPoolingQueryFactory(config)

    config["inputs"]["x"]["in_features"] = 5
    factory.config["inputs"]["x"]["in_features"] = 7

    assert factory.build().encoders["x"].in_features == 2


def test_conditioned_pooling_query_builds_encoders_from_dotted_paths():
    factory = ConditionedPoolingQueryFactory(FACTORY_CONFIGS["conditioned_pooling_query"])

    query = factory.build()

    assert isinstance(query.encoders["obj_patch"], TokenReduction)
    assert isinstance(query.encoders["obj_patch"].module, PatchEmbedding)
    assert query.encoders["obj_patch"].reduce == "mean"
    assert isinstance(query.encoders["obj_patch_location"], nn.Linear)
    assert query(2, **_conditioning()).shape == (2, 1, EMBED_DIM)


def test_conditioned_pooling_query_accepts_any_nn_module_encoder():
    factory = ConditionedPoolingQueryFactory({
        "embed_dim": EMBED_DIM,
        "inputs": {
            "sequence": {
                "module": "torch.nn.Embedding",
                "num_embeddings": 5,
                "embedding_dim": EMBED_DIM,
                "reduce": "max",
            },
        },
    })

    query = factory.build()

    assert query(2, sequence=torch.tensor([[0, 1, 2], [3, 4, 0]])).shape == (2, 1, EMBED_DIM)


@pytest.mark.parametrize(
    "inputs, error, match",
    [
        ({}, ValueError, "inputs must be a non-empty mapping"),
        ({"x": "nope"}, ValueError, "inputs.x must be a mapping"),
        ({"x": {"in_features": 2}}, ValueError, "inputs.x.module must be a fully-qualified"),
        ({"x": {"module": "Linear"}}, ValueError, "fully-qualified dotted path"),
        ({"x": {"module": "no_such_module_xyz.Thing"}}, ImportError, "could not be imported"),
        ({"x": {"module": "torch.nn.NotAReal"}}, ValueError, "has no attribute"),
        ({"x": {"module": "torch.optim.SGD"}}, TypeError, "does not resolve to an nn.Module"),
        ({"x": {"module": "torch.nn.Linear"}}, TypeError, "Invalid .*inputs.x config"),
        (
            {"x": {"module": "torch.nn.Linear", "in_features": 2, "out_features": 3}},
            ValueError,
            "must produce 8 features",
        ),
    ],
)
def test_conditioned_pooling_query_reports_bad_input_specs(inputs, error, match):
    with pytest.raises(error, match=match):
        ConditionedPoolingQueryFactory({"embed_dim": EMBED_DIM, "inputs": inputs})


def test_custom_module_factory_subclass():
    class Identity(nn.Module):
        embed_dim = EMBED_DIM

        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = scale

    class IdentityFactory(ModuleFactory):
        module_class = Identity

    assert IdentityFactory({"scale": 2.0}).build().scale == 2.0


# -- composite model ----------------------------------------------------


def test_patch_transformer_config_only_accepts_class_token():
    assert not PatchTransformer({}).has_class_token
    assert not PatchTransformer(None).has_class_token
    assert not PatchTransformer({"class_token": False}).has_class_token
    assert PatchTransformer({"class_token": True}).has_class_token
    assert PatchTransformer({"class_token": {"init_std": 0.1}}).has_class_token
    with pytest.raises(ValueError, match="only accepts 'class_token'"):
        PatchTransformer({"embed_dim": 8})
    with pytest.raises(TypeError, match="class_token must be a boolean or a mapping"):
        PatchTransformer({"class_token": "yes"})
    with pytest.raises(TypeError, match="Invalid patch_transformer.class_token config"):
        PatchTransformer({"class_token": {"std": 0.1}})
    with pytest.raises(ValueError, match="init_std"):
        PatchTransformer({"class_token": {"init_std": -1}})


def test_class_token_is_prepended_after_positional_embedding():
    model = PatchTransformer({"class_token": True})
    model.setup(_fake_session_for(PatchTransformer))

    assert isinstance(model.class_token, ClassToken)
    assert model.class_token.embed_dim == EMBED_DIM
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 5, EMBED_DIM)
    # The class token has no position, so resizing the positional table
    # for a larger grid still works.
    assert model(torch.randn(2, 3, 12, 16)).shape == (2, 13, EMBED_DIM)
    assert "class_token.token" in dict(model.named_parameters())


def test_class_token_widens_the_padding_mask_for_encoder_and_pooling():
    model = PooledPatchTransformer({"class_token": True})
    model.setup(_fake_session_for(PooledPatchTransformer))
    seen_masks = []
    model.sequence_encoder.register_forward_hook(
        lambda module, args, kwargs, output: seen_masks.append(kwargs["key_padding_mask"]),
        with_kwargs=True,
    )
    model.pooling.register_forward_hook(
        lambda module, args, kwargs, output: seen_masks.append(kwargs["key_padding_mask"]),
        with_kwargs=True,
    )
    mask = torch.tensor([[False, False, False, True], [False, True, True, True]])

    output = model.eval()(torch.randn(2, 3, 8, 8), key_padding_mask=mask)

    assert output.shape == (2, 2, EMBED_DIM)
    expected = torch.cat([torch.zeros(2, 1, dtype=torch.bool), mask], dim=1)
    assert len(seen_masks) == 2
    for seen in seen_masks:
        torch.testing.assert_close(seen, expected)


def test_class_token_is_part_of_saved_state():
    model = PatchTransformer({"class_token": True})
    model.setup(_fake_session_for(PatchTransformer))
    state = pickle.loads(pickle.dumps(model.get_state()))
    assert "class_token" in state["modules"]

    restored = PatchTransformer({"class_token": True})
    restored.set_state(state)
    torch.testing.assert_close(restored.class_token.token, model.class_token.token)
    images = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(restored.eval()(images), model.eval()(images))

    with pytest.raises(ValueError, match="state has a class token but class_token is disabled"):
        PatchTransformer({}).set_state(state)
    without = PatchTransformer({})
    without.setup(_fake_session_for(PatchTransformer))
    with pytest.raises(ValueError, match="state lacks a class token but class_token is enabled"):
        PatchTransformer({"class_token": True}).set_state(without.get_state())


def test_patch_transformer_is_unusable_before_setup():
    model = PatchTransformer({})
    assert not model.is_built
    assert model.get_state() is None
    assert list(model.parameters()) == []
    with pytest.raises(RuntimeError, match="has no blocks yet"):
        model(torch.randn(1, 3, 8, 8))


def test_patch_transformer_builds_blocks_in_setup():
    model = PatchTransformer({})
    model.setup(_fake_session_for(PatchTransformer))

    assert model.is_built
    assert model.embed_dim == EMBED_DIM
    assert isinstance(model.patch_embedding, PatchEmbedding)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 4, EMBED_DIM)
    # Larger inputs resize the learned positional table.
    assert model(torch.randn(2, 3, 12, 16)).shape == (2, 12, EMBED_DIM)


def test_pooled_patch_transformer_forwards_conditioning_to_query():
    model = PooledPatchTransformer({})
    session = _fake_session_for(
        PooledPatchTransformer,
        pooling_query=ConditionedPoolingQueryFactory(FACTORY_CONFIGS["conditioned_pooling_query"]),
    )
    model.setup(session)

    assert model(torch.randn(2, 3, 8, 8), **_conditioning()).shape == (2, 1, EMBED_DIM)
    with pytest.raises(TypeError, match="expects inputs"):
        model(torch.randn(2, 3, 8, 8))


def test_setup_rejects_blocks_with_different_embed_dims():
    model = PatchTransformer({})
    session = _fake_session_for(
        PatchTransformer,
        sequence_encoder=TorchTransformerEncoderFactory({
            **FACTORY_CONFIGS["torch_transformer_encoder"],
            "embed_dim": 16,
        }),
    )
    with pytest.raises(ValueError, match="disagree on embed_dim"):
        model.setup(session)
    assert not model.is_built


def test_setup_rejects_roles_not_bound_to_factories():
    model = PatchTransformer({})
    session = _fake_session_for(PatchTransformer, sequence_encoder=object())
    with pytest.raises(TypeError, match="'sequence_encoder' to be a ModuleFactory"):
        model.setup(session)


def test_restored_state_is_kept_by_setup():
    model = PatchTransformer({})
    model.setup(_fake_session_for(PatchTransformer))
    state = pickle.loads(pickle.dumps(model.get_state()))

    restored = PatchTransformer({})
    restored.set_state(state)
    restored_encoder = restored.sequence_encoder
    restored.setup(_fake_session_for(PatchTransformer))

    assert restored.sequence_encoder is restored_encoder
    images = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(restored.eval()(images), model.eval()(images))


def test_set_state_rejects_mismatched_blocks():
    with pytest.raises(ValueError, match="expects blocks"):
        PatchTransformer({}).set_state({"modules": {"patch_embedding": nn.Identity()}})


def test_composite_model_pickles_directly():
    model = PooledPatchTransformer({})
    model.setup(_fake_session_for(PooledPatchTransformer))

    restored = pickle.loads(pickle.dumps(model))

    images = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(restored.eval()(images), model.eval()(images))


# -- real sessions ------------------------------------------------------


def _register_sgd_step():
    class _SgdStep(Step):
        def __init__(self, config):
            self.losses = []

        def run(self, session):
            model = session.get_resource("model")
            model.train()
            output = model(torch.ones(2, 3, 8, 8), **_conditioning())
            loss = output.pow(2).mean()
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter -= 0.1 * parameter.grad
            self.losses.append(loss.item())

    return requires_resource("model")(step("transformer_sgd", session_type="training")(_SgdStep))


def test_training_session_trains_checkpoints_and_restores_the_model(tmp_path):
    _register_sgd_step()
    session = TrainingSession(_pooled_config(
        tmp_path,
        max_iterations=2,
        extra={"transformer_sgd": {}},
    ))
    initial_state = session.get_state()["components_state"]
    assert initial_state["pooled_patch_transformer"]["state"] is None

    with session:
        model = session.get_resource("model")
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for _ in session:
            pass

    assert model.is_built
    assert any(
        not torch.equal(before[name], p) for name, p in model.named_parameters()
    )
    assert {name.split(".")[0] for name, _ in model.named_parameters()} == set(
        PooledPatchTransformer.block_roles
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(session, checkpoint_path)
    restored_session = torch.load(checkpoint_path, weights_only=False)
    restored_model = restored_session.get_resource("model")

    # The checkpoint is self-contained: no setup() needed to use the model.
    assert restored_model.is_built
    images = torch.randn(2, 3, 8, 8)
    conditioning = _conditioning()
    torch.testing.assert_close(
        restored_model.eval()(images, **conditioning),
        model.eval()(images, **conditioning),
    )


def test_session_state_round_trip_before_setup_builds_in_worker(tmp_path):
    session = TrainingSession(_pooled_config(
        tmp_path,
        positional="sinusoidal_positional_embedding_2d",
        query="learned_pooling_query",
    ))

    worker_session = TrainingSession.from_state(
        pickle.loads(pickle.dumps(session.get_state()))
    )
    with worker_session:
        model = worker_session.get_resource("model")
        assert model.is_built
        assert isinstance(model.positional_embedding, SinusoidalPositionalEmbedding2D)
        assert model(torch.randn(3, 3, 8, 8)).shape == (3, 2, EMBED_DIM)


def test_patch_transformer_without_pooling_in_a_session(tmp_path):
    session = TrainingSession(_pooled_config(tmp_path, model="patch_transformer"))
    with session:
        model = session.get_resource("model")
        assert type(model) is PatchTransformer
        assert model(torch.randn(1, 3, 8, 8)).shape == (1, 4, EMBED_DIM)


@pytest.mark.parametrize("class_token", [False, True])
def test_trained_model_loads_transformer_checkpoint_for_analysis(tmp_path, class_token):
    config = _pooled_config(tmp_path)
    config["pooled_patch_transformer"] = {"class_token": class_token}
    session = TrainingSession(config)
    with session:
        trained = session.get_resource("model")
    checkpoint_path = tmp_path / "training.pt"
    torch.save(session, checkpoint_path)

    analysis = AnalysisSession({
        "session_config": _session_config(tmp_path / "analysis"),
        "trained_model": {"model_checkpoint_path": str(checkpoint_path)},
    })
    with analysis:
        model = analysis.get_resource("trained_model").model
        assert isinstance(analysis.get_resource("trained_model"), TrainedModel)
        assert model.has_class_token is class_token
        images = torch.randn(2, 3, 8, 8)
        conditioning = _conditioning()
        torch.testing.assert_close(
            model(images, **conditioning),
            trained.eval()(images, **conditioning),
        )
