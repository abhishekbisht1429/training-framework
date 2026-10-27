"""Fine-tuning: a model from another run, partly frozen, and a new head.

A pretrained model comes in through `import_components`, which makes it --
and everything it was wired to -- components of this session. Two resources
build on that. `module_part` takes a part of a module resource (an encoder
out of a whole pretrained model, say) and calls one of its methods.
`backbone_with_head` composes whatever fills the `backbone` and `head`
roles, freezes part of the backbone, and runs `head(backbone(x))`; bound as
`model`, `ddp`, `forward` and `optimizer` use it like any other model.
`fine_tuned_model` is its deprecated former name, kept for old
configurations and checkpoints.

The backbone need not come from another run -- any resource that is an
`nn.Module` can fill the role -- and the head is always the user's own
resource, which declares whatever it needs (a `dataset` for its class
count, say) itself.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ModuleResource,
    Resource,
    requires_resource,
    resource,
    role,
)
from training_framework.components.builtin.parameter_patterns import (
    check_patterns_match,
    matches_any,
    parameter_patterns,
)

if TYPE_CHECKING:
    from training_framework.session import Session


role(
    "backbone",
    Resource,
    description=(
        "an nn.Module resource whose output the head reads, e.g. an imported "
        "model or a module_part of one"
    ),
)
role(
    "head",
    Resource,
    description="an nn.Module resource applied to the backbone's output",
)
role(
    "source",
    Resource,
    description="the nn.Module resource a module_part takes a part of",
)


_MISSING = object()


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


# -- module_part ----------------------------------------------------------


@dataclass
class ModulePartConfig:
    submodule: Any = ""
    method: Any = "forward"

    def __post_init__(self):
        if not isinstance(self.submodule, str):
            raise ValueError(
                "submodule must be an attribute path such as "
                f"'encoder.blocks', or '' for the whole source; got "
                f"{self.submodule!r}"
            )
        if self.submodule and not all(self.submodule.split(".")):
            raise ValueError(
                f"submodule {self.submodule!r} has an empty attribute name"
            )
        self.method = _non_empty_string(self.method, "method")


@requires_resource("source")
@resource("module_part", session_type=TRAINING_SESSION_TYPE)
@resource("module_part", session_type=ANALYSIS_SESSION_TYPE)
class ModulePart(ModuleResource):
    """A part of the module resource bound to `source`, and one method of it.

    `submodule` is an attribute path inside the source (`''`, the default,
    is the whole source); the part is held under the attribute `module`, so
    its parameter names start with `module.`. Calling this resource calls
    the part's `method` (default `forward`).

    The part stays the source's: the source trains, saves and restores its
    weights, and this component owns none. What the source holds outside
    the part stays in the session, unused by anything that only calls this
    component.
    """

    config_schema = ModulePartConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        name = self._component_name()
        source = self.get_dependency("source")
        if not isinstance(source, nn.Module):
            raise TypeError(
                f"{name} role 'source' is filled by {type(source).__name__}, "
                "which is not an nn.Module"
            )
        self.module = self._select_submodule(source)
        if not callable(getattr(self.module, self._cfg.method, None)):
            raise ValueError(
                f"{name}.method: {type(self.module).__name__} has no method "
                f"{self._cfg.method!r}"
            )

    def _select_submodule(self, source: nn.Module) -> nn.Module:
        name = self._component_name()
        current: Any = source
        walked = getattr(source, "name", "source")
        for attribute in self._cfg.submodule.split(".") if self._cfg.submodule else ():
            try:
                child = getattr(current, attribute)
            except AttributeError:
                children = (
                    [child_name for child_name, _ in current.named_children()]
                    if isinstance(current, nn.Module) else []
                )
                raise ValueError(
                    f"{name}.submodule: {walked} ({type(current).__name__}) "
                    f"has no attribute {attribute!r}; its submodules are "
                    f"{children}"
                ) from None
            walked = f"{walked}.{attribute}"
            current = child
        if not isinstance(current, nn.Module):
            raise TypeError(
                f"{name}: {walked} is a {type(current).__name__}, not an "
                "nn.Module"
            )
        return current

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.module, self._cfg.method)(*args, **kwargs)


# -- backbone_with_head (and fine_tuned_model, its former name) -----------


@dataclass
class FineTunedModelConfig:
    frozen: Any = ()
    frozen_eval: Any = None
    backbone_output: Any = None

    def __post_init__(self):
        # Only an empty list means "nothing frozen": `false`, `0` or a key
        # left without a value is a mistake, not a request to train it all.
        if isinstance(self.frozen, (list, tuple)) and not self.frozen:
            self.frozen = ()
        else:
            self.frozen = tuple(parameter_patterns(self.frozen, "frozen"))
        if self.frozen_eval is not None and not isinstance(self.frozen_eval, bool):
            raise ValueError(
                f"frozen_eval must be true or false; got {self.frozen_eval!r}"
            )
        if self.frozen_eval and not self.frozen:
            raise ValueError(
                "frozen_eval is true but nothing is frozen; list the frozen "
                "parameters under `frozen`"
            )
        if self.frozen_eval is None:
            # Frozen means frozen: BatchNorm statistics and dropout in a
            # frozen block change nothing about its weights, but they do
            # change what it computes.
            self.frozen_eval = bool(self.frozen)
        selector = self.backbone_output
        if not (
                selector is None
                or (isinstance(selector, int) and not isinstance(selector, bool))
                or (isinstance(selector, str) and selector)
        ):
            raise ValueError(
                "backbone_output must be an index, a key or attribute name, "
                f"or null; got {selector!r}"
            )


#: How `backbone_with_head` reduces `(B, N, D)` backbone tokens to `(B, D)`.
POOLING_MODES = ("cls", "mean")


@dataclass
class BackboneWithHeadConfig(FineTunedModelConfig):
    pool: Any = None

    def __post_init__(self):
        super().__post_init__()
        if self.pool is not None and self.pool not in POOLING_MODES:
            raise ValueError(
                f"pool must be one of {list(POOLING_MODES)} or null; got "
                f"{self.pool!r}"
            )
        if self.pool is not None and self.backbone_output is not None:
            raise ValueError(
                "pool reads the backbone's tokens through encode_with_mask, "
                "not its forward output, so backbone_output (which selects "
                "from that output) cannot be combined with it"
            )


@requires_resource("backbone")
@requires_resource("head")
@resource("backbone_with_head", session_type=TRAINING_SESSION_TYPE)
@resource("backbone_with_head", session_type=ANALYSIS_SESSION_TYPE)
class BackboneWithHead(ModuleResource):
    """`head(pool(backbone(x)))`, with part of the backbone frozen.

    Both parts are prerequisites attached as the submodules `backbone` and
    `head`, so parameter names read `backbone.<...>` and `head.<...>` (what
    `optimizer.param_groups` patterns match) and each part checkpoints its
    own weights; this component owns none.

    Config:
    - `frozen`: glob patterns over the backbone's parameter names (relative
      to the backbone; a `module_part`'s names start with `module.`).
      Each must match. Matching parameters get `requires_grad=False` when
      this component is built -- before `ddp` wraps the model -- so they get
      no gradient, cost no backward pass, and are never updated. Re-applied
      on every build, since `requires_grad` is not part of any state. For
      freezing that ends at some iteration, use `freeze_gradients`.
    - `frozen_eval` (default: true when anything is frozen): every backbone
      submodule whose parameters are all frozen stays in eval mode, so its
      BatchNorm statistics and dropout do not change what it computes.
    - `backbone_output`: which part of the backbone's output the head gets:
      an index into a tuple/list, a key of a mapping (a Hugging Face output),
      or an attribute of another object. Default: the whole output.
    - `pool` (default none): reduce the backbone's tokens to `(B, D)`
      before the head. The tokens come from `encode_with_mask` -- the
      token-encoder contract `patch_transformer` follows: `(B, N, D)`
      tokens, the class token first exactly when `has_class_token`, and the
      padding mask that covers them -- never from `forward`, which a
      subclass may change (`pooled_patch_transformer` pools there). `cls`
      takes the class token; `mean` averages the other tokens, leaving out
      padded ones. Not combined with `backbone_output`.
    """

    config_schema = BackboneWithHeadConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        for role_name in ("backbone", "head"):
            part = self.get_dependency(role_name)
            if not isinstance(part, nn.Module):
                raise TypeError(
                    f"{self._component_name()} role {role_name!r} is filled "
                    f"by {type(part).__name__}, which is not an nn.Module"
                )
            setattr(self, role_name, part)
        self._check_pool()
        self._frozen_module_names = self._freeze()
        self.train(self.training)

    def _pool_mode(self) -> str | None:
        return self._cfg.pool

    def _check_pool(self) -> None:
        mode = self._pool_mode()
        if mode is None:
            return
        has_class_token = getattr(self.backbone, "has_class_token", None)
        encode_with_mask = getattr(self.backbone, "encode_with_mask", None)
        if not isinstance(has_class_token, bool) or not callable(encode_with_mask):
            raise ValueError(
                f"{self._component_name()}.pool is {mode!r}, but the backbone "
                f"({type(self.backbone).__name__}) is not a token encoder: "
                "pooling needs encode_with_mask (tokens and their padding "
                "mask) and a boolean has_class_token, as patch_transformer "
                "has, so the class token can be told apart from the others"
            )
        if mode == "cls" and not has_class_token:
            raise ValueError(
                f"{self._component_name()}.pool is 'cls', but the backbone has "
                "no class token; set class_token: true on it, or use pool: mean"
            )

    def _freeze(self) -> tuple[str, ...]:
        """Freeze the matching backbone parameters; return the backbone
        submodules left with no trainable parameter, outermost only."""
        patterns = self._cfg.frozen
        if not patterns:
            return ()
        named = list(self.backbone.named_parameters())
        check_patterns_match(
            patterns,
            [parameter_name for parameter_name, _ in named],
            f"{self._component_name()}.frozen",
        )
        frozen = set()
        for parameter_name, parameter in named:
            if matches_any(parameter_name, patterns):
                parameter.requires_grad_(False)
                frozen.add(id(parameter))

        def fully_frozen(module: nn.Module) -> bool:
            parameters = list(module.parameters())
            return bool(parameters) and all(
                id(parameter) in frozen for parameter in parameters
            )

        if fully_frozen(self.backbone):
            return ("",)
        names: list[str] = []

        def visit(module: nn.Module, prefix: str) -> None:
            for child_name, child in module.named_children():
                path = f"{prefix}{child_name}"
                if fully_frozen(child):
                    names.append(path)
                else:
                    visit(child, f"{path}.")

        visit(self.backbone, "")
        return tuple(names)

    def train(self, mode: bool = True) -> "BackboneWithHead":
        super().train(mode)
        if mode and self._cfg.frozen_eval:
            for module_name in self._frozen_module_names:
                self.backbone.get_submodule(module_name).eval()
        return self

    def setup(self, session: "Session") -> None:
        super().setup(session)
        # A prerequisite's own setup may have changed modes; settle them.
        self.train(self.training)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._pool_mode() is None:
            return self.head(self._select(self.backbone(*args, **kwargs)))
        tokens, mask = self.backbone.encode_with_mask(*args, **kwargs)
        return self.head(self._pool(tokens, mask))

    def _pool(self, tokens: Tensor, mask: Tensor | None) -> Tensor:
        """Reduce `(B, N, D)` tokens to `(B, D)`; `mask` (`(B, N)`, true
        where a token is padding) is what `encode_with_mask` returned."""
        name = f"{self._component_name()}.pool"
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise TypeError(
                f"{name} reduces (B, N, D) tokens, but encode_with_mask "
                "returned "
                + (
                    f"a tensor of shape {tuple(tokens.shape)}"
                    if isinstance(tokens, Tensor)
                    else f"a {type(tokens).__name__}"
                )
            )
        if self._pool_mode() == "cls":
            return tokens[:, 0]
        if self.backbone.has_class_token:
            tokens = tokens[:, 1:]
            mask = mask[:, 1:] if mask is not None else None
        if mask is None:
            return tokens.mean(dim=1)
        if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
            raise TypeError(
                f"{name} needs a boolean (B, N) padding mask over the tokens; "
                f"got {mask.dtype} {tuple(mask.shape)}"
            )
        kept = (~mask).unsqueeze(-1).to(tokens.dtype)
        counts = kept.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError(f"{name} is 'mean', but a sample has every token masked")
        return (tokens * kept).sum(dim=1) / counts

    def _select(self, output: Any) -> Any:
        selector = self._cfg.backbone_output
        if selector is None:
            return output
        name = f"{self._component_name()}.backbone_output"
        returned = type(output).__name__
        if isinstance(selector, int):
            if not isinstance(output, Sequence) or isinstance(output, str):
                raise TypeError(
                    f"{name} is the index {selector}, but the backbone "
                    f"returned a {returned}"
                )
            if not -len(output) <= selector < len(output):
                raise IndexError(
                    f"{name} is the index {selector}, but the backbone "
                    f"returned {len(output)} values"
                )
            return output[selector]
        if isinstance(output, Mapping):
            if selector not in output:
                raise KeyError(
                    f"{name} is {selector!r}, but the backbone returned a "
                    f"{returned} with keys {list(output)}"
                )
            return output[selector]
        missing = TypeError(
            f"{name} is {selector!r}, but the backbone returned a "
            f"{returned} without such a key or attribute"
        )
        if isinstance(output, Tensor):
            # Never looked up: a tensor has many attributes (`T`, `shape`),
            # none of them a part of the output, and a subclass's property
            # would run.
            raise missing
        # Read once: a property could compute, or change, on a second read.
        value = getattr(output, selector, _MISSING)
        if value is _MISSING:
            raise missing
        return value


@resource("fine_tuned_model", session_type=TRAINING_SESSION_TYPE)
@resource("fine_tuned_model", session_type=ANALYSIS_SESSION_TYPE)
class FineTunedModel(BackboneWithHead):
    """Deprecated: `backbone_with_head` under its former name.

    Kept so configurations and checkpoints that name `fine_tuned_model`
    still build and restore: same roles, same configuration (no `pool`),
    same parameter names and state. A run saved under this name resumes
    under it; new configurations use `backbone_with_head`.
    """

    config_schema = FineTunedModelConfig

    def __init__(self, config: Mapping | None = None) -> None:
        warnings.warn(
            "fine_tuned_model is deprecated; use backbone_with_head, which "
            "takes the same configuration (and adds pool)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)

    def _pool_mode(self) -> str | None:
        return None


__all__ = ["BackboneWithHead", "FineTunedModel", "ModulePart", "POOLING_MODES"]
