"""The networks: `projection_head`, `backbone_with_head` (with `pool`, and
its deprecated name `fine_tuned_model`) and `multi_view`."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional

from tests.integration.integration_ssl_components import EMBED_DIM, dino_config, vit
from tests.test_utils import build_session, make_config, resource_named
from training_framework.components import ModuleResource, resource
from training_framework.components.builtin import (
    BackboneWithHead,
    Checkpointer,
    FineTunedModel,
)
from training_framework.session import TrainingSession

from .helpers import quiet_session


class PlainTokens(ModuleResource):
    """A backbone returning tokens without saying if it has a class token."""

    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(3, EMBED_DIM)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture(autouse=True)
def _plain_tokens():
    resource("ssl_plain_tokens", overwrite=True)(PlainTokens)


def _head(tmp_path, **config):
    session = build_session(tmp_path, {"projection_head": {"in_dim": 6, **config}})
    return resource_named(session, "projection_head")


# -- projection_head ----------------------------------------------------------------


def test_a_projection_head_is_an_mlp_with_norm_and_activation_between_layers(tmp_path):
    head = _head(tmp_path, hidden_dim=10, out_dim=4, num_layers=3, norm="batch",
                 activation="relu")

    assert [type(layer) for layer in head.mlp] == [
        nn.Linear, nn.BatchNorm1d, nn.ReLU,
        nn.Linear, nn.BatchNorm1d, nn.ReLU,
        nn.Linear,
    ]
    assert head(torch.randn(5, 6)).shape == (5, 4)
    assert (head.in_embed_dim, head.out_embed_dim) == (6, 4)


def test_one_layer_is_a_single_linear_layer(tmp_path):
    head = _head(tmp_path, out_dim=4, num_layers=1)

    assert [type(layer) for layer in head.mlp] == [nn.Linear]


def test_the_default_initialization_is_a_truncated_normal_and_null_keeps_torchs(tmp_path):
    dino = _head(tmp_path / "a", init_std=0.02)
    torch_default = _head(tmp_path / "b", init_std=None)

    assert torch.count_nonzero(dino.mlp[0].bias) == 0
    assert dino.mlp[0].weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert torch.count_nonzero(torch_default.mlp[0].bias) > 0


def test_prototypes_normalize_then_project_with_a_frozen_unit_gain(tmp_path):
    head = _head(tmp_path, out_dim=4, prototypes={"num": 9})
    features = torch.randn(3, 6)

    output = head(features)

    gain = head.last_layer.gain
    assert output.shape == (3, 9) and head.out_embed_dim == 9
    assert torch.all(gain == 1) and not gain.requires_grad
    torch.testing.assert_close(
        output, head.last_layer(functional.normalize(head.mlp(features), dim=-1)),
    )
    # The input's scale cannot reach the prototypes: the bottleneck is
    # normalized first.
    bottleneck = head.mlp(features)
    torch.testing.assert_close(
        head.last_layer(functional.normalize(3 * bottleneck, dim=-1)), output,
    )


def test_the_prototype_layer_is_torchs_weight_norm(tmp_path):
    head = _head(tmp_path, out_dim=4, prototypes={"num": 9})
    reference = torch.nn.utils.parametrizations.weight_norm(
        torch.nn.Linear(4, 9, bias=False),
    )
    with torch.no_grad():
        reference.parametrizations.weight.original0.copy_(head.last_layer.gain)
        reference.parametrizations.weight.original1.copy_(head.last_layer.direction)
    features = torch.randn(3, 4)

    torch.testing.assert_close(head.last_layer(features), reference(features))


def test_an_unfrozen_last_layer_gain_trains(tmp_path):
    head = _head(tmp_path, prototypes={"num": 9, "norm_last_layer": False})

    assert head.last_layer.gain.requires_grad


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"norm": "group"}, "norm must be one of"),
        ({"activation": "swish"}, "activation must be one of"),
        ({"num_layers": 0}, "num_layers must be a positive integer"),
        ({"prototypes": {"count": 3}}, r"unknown keys \['count'\]"),
        ({"prototypes": {}}, "prototypes needs num"),
        ({"prototypes": {"num": 3, "norm_last_layer": "yes"}}, "true or false"),
        ({"prototypes": 3}, "null or a mapping"),
    ],
)
def test_invalid_head_configuration_is_reported(tmp_path, config, message):
    with pytest.raises(ValueError, match=message):
        _head(tmp_path, **config)


def test_a_head_given_tokens_says_to_pool_them(tmp_path):
    head = _head(tmp_path)

    with pytest.raises(ValueError, match=r"expects \(B, 6\).*needs pooling first"):
        head(torch.randn(2, 5, 6))


# -- backbone_with_head ----------------------------------------------------------


def _pooled_model(tmp_path, *, pool, class_token=True, name="backbone_with_head"):
    components = {
        **vit("x"),
        "projection_head": {"in_dim": EMBED_DIM, "out_dim": 4, "num_layers": 1},
        name: {
            **({"pool": pool} if pool is not None else {}),
            "dependencies_role_bindings": {
                "backbone": "patch_transformer#x", "head": "projection_head",
            },
        },
    }
    components["patch_transformer#x"]["class_token"] = class_token
    session = build_session(tmp_path, components)
    return resource_named(session, name)


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_pool_reduces_the_tokens_before_the_head(tmp_path, pool):
    model = _pooled_model(tmp_path, pool=pool)
    images = torch.randn(2, 3, 32, 32)

    tokens = model.backbone(images)
    pooled = tokens[:, 0] if pool == "cls" else tokens[:, 1:].mean(dim=1)

    torch.testing.assert_close(model(images), model.head(pooled))


def test_mean_pooling_without_a_class_token_averages_every_token(tmp_path):
    model = _pooled_model(tmp_path, pool="mean", class_token=False)
    images = torch.randn(2, 3, 32, 32)

    torch.testing.assert_close(
        model(images), model.head(model.backbone(images).mean(dim=1)),
    )


def test_cls_pooling_needs_a_class_token(tmp_path):
    with pytest.raises(ValueError, match="pool is 'cls', but the backbone has no class token"):
        _pooled_model(tmp_path, pool="cls", class_token=False)


def test_pooling_needs_a_token_encoder_backbone(tmp_path):
    with pytest.raises(ValueError, match="is not a token encoder.*encode_with_mask"):
        build_session(tmp_path, {
            "ssl_plain_tokens": {},
            "projection_head": {"in_dim": EMBED_DIM},
            "backbone_with_head": {
                "pool": "mean",
                "dependencies_role_bindings": {
                    "backbone": "ssl_plain_tokens", "head": "projection_head",
                },
            },
        })


def test_mean_pooling_leaves_out_padded_tokens(tmp_path):
    model = _pooled_model(tmp_path, pool="mean")
    images = torch.randn(2, 3, 32, 32)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[0, 10:] = True                        # sample 0: 6 patches padded

    tokens, mask = model.backbone.encode_with_mask(images, padding)
    patches = tokens[:, 1:]
    expected = torch.stack([patches[0, :10].mean(dim=0), patches[1].mean(dim=0)])

    torch.testing.assert_close(
        model(images, key_padding_mask=padding), model.head(expected),
    )


def test_pooling_a_pooled_patch_transformer_reads_its_tokens_not_its_queries(tmp_path):
    components = {
        **vit("x"),
        "learned_pooling_query": {"embed_dim": EMBED_DIM, "num_queries": 3},
        "attention_pooling": {"embed_dim": EMBED_DIM, "num_heads": 2},
        "pooled_patch_transformer": {
            "class_token": True,
            "dependencies_role_bindings": {
                "patch_embedding": "conv_patch_embedding#x",
                "positional_embedding": "learned_positional_embedding_2d#x",
                "sequence_encoder": "torch_transformer_encoder#x",
                "pooling": "attention_pooling",
                "pooling_query": "learned_pooling_query",
            },
        },
        "projection_head": {"in_dim": EMBED_DIM, "out_dim": 4, "num_layers": 1},
        "backbone_with_head": {
            "pool": "cls",
            "dependencies_role_bindings": {
                "backbone": "pooled_patch_transformer", "head": "projection_head",
            },
        },
    }
    model = resource_named(build_session(tmp_path, components), "backbone_with_head")
    images = torch.randn(2, 3, 32, 32)

    class_token = model.backbone.encode(images)[:, 0]
    first_query = model.backbone(images)[:, 0]

    torch.testing.assert_close(model(images), model.head(class_token))
    assert not torch.allclose(model.head(class_token), model.head(first_query))


def test_pool_and_backbone_output_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="backbone_output.*cannot be combined"):
        build_session(tmp_path, {
            **vit("x"),
            "projection_head": {"in_dim": EMBED_DIM},
            "backbone_with_head": {
                "pool": "cls", "backbone_output": 0,
                "dependencies_role_bindings": {
                    "backbone": "patch_transformer#x", "head": "projection_head",
                },
            },
        })


def test_an_unknown_pool_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"pool must be one of \['cls', 'mean'\]"):
        _pooled_model(tmp_path, pool="max")


def test_without_pool_the_backbone_output_reaches_the_head_unchanged(tmp_path):
    model = _pooled_model(tmp_path, pool=None)
    images = torch.randn(2, 3, 32, 32)

    with pytest.raises(ValueError, match="needs pooling first"):
        model(images)


# -- fine_tuned_model, the former name ---------------------------------------------


def test_fine_tuned_model_still_builds_and_warns_that_it_is_deprecated(tmp_path):
    with pytest.warns(DeprecationWarning, match="use backbone_with_head"):
        legacy = _pooled_model(tmp_path / "a", pool=None, name="fine_tuned_model")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        current = _pooled_model(tmp_path / "b", pool=None)

    assert type(legacy) is FineTunedModel and isinstance(legacy, BackboneWithHead)
    assert type(current) is BackboneWithHead
    assert legacy.name == "fine_tuned_model" and current.name == "backbone_with_head"
    # Same construction (seeded), same parameter names, same function.
    assert list(legacy.state_dict()) == list(current.state_dict())
    for (name, mine), (_, theirs) in zip(
            legacy.state_dict().items(), current.state_dict().items(),
    ):
        torch.testing.assert_close(mine, theirs, msg=name)


def test_fine_tuned_model_takes_its_former_configuration_only(tmp_path):
    with pytest.warns(DeprecationWarning), pytest.raises(Exception, match="pool"):
        _pooled_model(tmp_path, pool="cls", name="fine_tuned_model")


def test_a_fine_tuned_model_checkpoint_restores_under_its_name(tmp_path):
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        **vit("x"),
        "projection_head": {"in_dim": EMBED_DIM, "num_layers": 1},
        "fine_tuned_model": {
            "frozen": ["patch_embedding.*"],
            "dependencies_role_bindings": {
                "backbone": "patch_transformer#x", "head": "projection_head",
            },
        },
    })
    with pytest.warns(DeprecationWarning):
        session = TrainingSession(config)
        saved = resource_named(session, "fine_tuned_model")
        path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")
        restored = Checkpointer.load_component(path, "fine_tuned_model")

    assert type(restored) is FineTunedModel
    assert not restored.backbone.patch_embedding.projection.weight.requires_grad
    for (name, mine), (_, theirs) in zip(
            saved.state_dict().items(), restored.state_dict().items(),
    ):
        torch.testing.assert_close(mine, theirs, msg=name)


# -- multi_view -----------------------------------------------------------------------


def _multi_view(tmp_path):
    session = quiet_session(dino_config(tmp_path))
    return resource_named(session, "multi_view#student")


def test_views_of_one_size_are_encoded_in_one_call_and_returned_in_order(tmp_path):
    model = _multi_view(tmp_path).eval()
    generator = torch.Generator().manual_seed(3)
    views = [
        torch.randn(2, 3, size, size, generator=generator)
        for size in (32, 32, 16, 16, 16)
    ]
    calls = []
    model.network.register_forward_hook(
        lambda module, args, output: calls.append(tuple(args[0].shape)),
    )

    output = model(views)

    assert calls == [(4, 3, 32, 32), (6, 3, 16, 16)]
    expected = torch.cat([model.network(view) for view in views])
    torch.testing.assert_close(output, expected)


def test_views_of_the_same_size_apart_from_each_other_are_separate_calls(tmp_path):
    model = _multi_view(tmp_path).eval()
    views = [torch.randn(2, 3, s, s) for s in (32, 16, 32)]
    calls = []
    model.network.register_forward_hook(
        lambda module, args, output: calls.append(args[0].shape[0]),
    )

    model(views)

    assert calls == [2, 2, 2]


def test_a_view_limit_uses_the_leading_views(tmp_path):
    session = quiet_session(dino_config(tmp_path))
    teacher = resource_named(session, "multi_view#teacher").eval()
    views = [torch.randn(2, 3, s, s) for s in (32, 32, 16, 16)]

    assert teacher.views == 2
    torch.testing.assert_close(teacher(views), teacher(views[:2]))
    with pytest.raises(ValueError, match="uses the first 2 views, but got 1"):
        teacher(views[:1])


@pytest.mark.parametrize(
    ("views", "message"),
    [
        (torch.randn(2, 3, 32, 32), "takes a list of view batches; got a Tensor"),
        ([], "non-empty list of tensors"),
        ([torch.randn(2, 3, 32, 32), torch.randn(3, 3, 32, 32)], "same batch"),
    ],
)
def test_multi_view_refuses_what_is_not_a_list_of_view_batches(tmp_path, views, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _multi_view(tmp_path)(views)
