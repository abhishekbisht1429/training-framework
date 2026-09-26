"""Fine-tuning: a module from another run, partly frozen, and a new head.

Two resources. `checkpoint_module` takes a module out of another run's
checkpoint and owns it from then on: it is trained and checkpointed here.
`fine_tuned_model` composes whatever fills the `backbone` and `head` roles,
freezes part of the backbone, and runs `head(backbone(x))`; bound as
`model`, `ddp`, `forward` and `optimizer` use it like any other model.

The backbone need not come from a checkpoint -- any resource that is an
`nn.Module` can fill the role -- and the head is always the user's own
resource, which declares whatever it needs (a `dataset` for its class
count, say) itself.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ComponentDependencyError,
    ModuleResource,
    Resource,
    requires_resource,
    resource,
    role,
)
from training_framework.components.builtin.checkpointing import Checkpointer
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
        "an nn.Module resource whose output the head reads, e.g. "
        "checkpoint_module"
    ),
)
role(
    "head",
    Resource,
    description="an nn.Module resource applied to the backbone's output",
)


_MISSING = object()


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


# -- checkpoint_module ----------------------------------------------------


@dataclass
class CheckpointModuleConfig:
    checkpoint: Any
    resource: Any = "model"
    submodule: Any = ""
    method: Any = "forward"

    def __post_init__(self):
        try:
            checkpoint = os.fspath(self.checkpoint)
        except TypeError as error:
            raise TypeError(
                f"checkpoint must be a path; got {self.checkpoint!r}"
            ) from error
        if not isinstance(checkpoint, str) or not checkpoint:
            raise TypeError(f"checkpoint must be a path; got {self.checkpoint!r}")
        self.checkpoint = checkpoint
        self.resource = _non_empty_string(self.resource, "resource")
        if not isinstance(self.submodule, str):
            raise ValueError(
                "submodule must be an attribute path such as "
                f"'encoder.blocks', or '' for the whole resource; got "
                f"{self.submodule!r}"
            )
        if self.submodule and not all(self.submodule.split(".")):
            raise ValueError(
                f"submodule {self.submodule!r} has an empty attribute name"
            )
        self.method = _non_empty_string(self.method, "method")


@resource("checkpoint_module")
class CheckpointModule(ModuleResource):
    """A module taken out of another run's checkpoint, trained from here on.

    `resource` names a resource of that run the way that run named it -- a
    role such as `model` resolves through its own bindings, or an instance
    such as `model#teacher` -- and `submodule` is an attribute path inside
    it (`''`, the default, keeps the whole resource). Only that module is
    kept, under the attribute `module` (as `DistributedDataParallel` does),
    so parts of the source model left out are neither trained nor saved.
    Calling this resource calls the module's `method` (default `forward`).

    The source checkpoint is read on every construction -- a fresh run, a
    resume, each worker -- for the module's architecture; a restored
    component's own state then replaces the weights. It must therefore stay
    where `checkpoint` points for as long as this run's checkpoints are used.
    """

    config_schema = CheckpointModuleConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        name = self._component_name()
        path = self._cfg.checkpoint
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}.checkpoint does not exist: {path}")
        try:
            loaded = Checkpointer.load_component(
                path, self._cfg.resource, map_location="cpu",
            )
        except (KeyError, ComponentDependencyError) as error:
            raise ValueError(
                f"{name}: checkpoint {path} has no single resource "
                f"{self._cfg.resource!r}: {error}"
            ) from error

        self.module = self._select_submodule(loaded)
        if not callable(getattr(self.module, self._cfg.method, None)):
            raise ValueError(
                f"{name}.method: {type(self.module).__name__} has no method "
                f"{self._cfg.method!r}"
            )
        try:
            self._check_child_ownership()
        except ComponentDependencyError as error:
            raise ComponentDependencyError(
                f"{name}: {self._source_label()} cannot be held as a plain "
                "module. Pick a part of it with `submodule`. "
                f"{error}"
            ) from error

    def _source_label(self) -> str:
        label = self._cfg.resource
        if self._cfg.submodule:
            label = f"{label}.{self._cfg.submodule}"
        return label

    def _select_submodule(self, loaded: Any) -> nn.Module:
        name = self._component_name()
        current = loaded
        walked = self._cfg.resource
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
      to the backbone; `checkpoint_module` names start with `module.`).
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


__all__ = ["CheckpointModule", "FineTunedModel"]
