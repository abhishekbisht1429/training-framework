"""Fine-tuning: a model from another run, partly frozen, and a new head.

A pretrained model comes in through `import_components`, which makes it --
and everything it was wired to -- components of this session. Two resources
build on that. `module_part` takes a part of a module resource (an encoder
out of a whole pretrained model, say) and calls one of its methods.
`fine_tuned_model` composes whatever fills the `backbone` and `head` roles,
freezes part of the backbone, and runs `head(backbone(x))`; bound as
`model`, `ddp`, `forward` and `optimizer` use it like any other model.

The backbone need not come from another run -- any resource that is an
`nn.Module` can fill the role -- and the head is always the user's own
resource, which declares whatever it needs (a `dataset` for its class
count, say) itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


# -- fine_tuned_model -----------------------------------------------------


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


@requires_resource("backbone")
@requires_resource("head")
@resource("fine_tuned_model", session_type=TRAINING_SESSION_TYPE)
@resource("fine_tuned_model", session_type=ANALYSIS_SESSION_TYPE)
class FineTunedModel(ModuleResource):
    """`head(backbone(x))`, with part of the backbone frozen.

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
    """

    config_schema = FineTunedModelConfig

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
        self._frozen_module_names = self._freeze()
        self.train(self.training)

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

    def train(self, mode: bool = True) -> "FineTunedModel":
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
        return self.head(self._select(self.backbone(*args, **kwargs)))

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


__all__ = ["FineTunedModel", "ModulePart"]
