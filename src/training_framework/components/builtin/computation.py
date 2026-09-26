"""Generic steps: read a batch, call a model, call a function.

Every training paradigm is made of these few operations wired differently --
one forward pass or a student and a teacher, one loss or several combined.
So these steps are used as many times as needed, through instances
(`forward#teacher`, `compute#kl`), and each declares the `iteration_context`
keys it reads and writes from its configuration. Those declarations are
what orders them: a step runs after the step that writes what it reads.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import torch
from torch import nn

from training_framework import functions as builtin_functions
from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    Step,
    requires_resource,
    step,
)
from training_framework.components.importing import import_object

if TYPE_CHECKING:
    from training_framework.session import Session


def _key(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{path} must be a non-empty iteration_context key; got {value!r}"
        )
    return value


def _parameter_names(mapping: Any, path: str) -> dict:
    """A mapping keyed by keyword-argument names, checked as such.

    Parameter names are passed as `**kwargs`, so a key that is not a
    non-empty string would only fail when the call runs -- inside a worker,
    mid-run -- instead of when the configuration is read.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{path} must be a mapping; got {mapping!r}")
    for parameter in mapping:
        if not isinstance(parameter, str) or not parameter:
            raise ValueError(
                f"{path} parameter names must be non-empty strings; got "
                f"{parameter!r}"
            )
    return dict(mapping)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _unique(keys: Sequence[str], path: str) -> tuple[str, ...]:
    repeated = sorted({key for key in keys if list(keys).count(key) > 1})
    if repeated:
        raise ValueError(f"{path} names {repeated} more than once")
    return tuple(keys)


def _returned(values: dict[str, Any]) -> Any:
    """`values` as a step returns them: one output is the value itself,
    several a mapping by output name."""
    if len(values) == 1:
        return next(iter(values.values()))
    return values


def _to_device(value: Any, device: torch.device, non_blocking: bool) -> Any:
    """Move every tensor in a batch to `device`, leaving the rest alone."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: _to_device(item, device, non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(
            _to_device(item, device, non_blocking) for item in value
        ))
    if isinstance(value, (list, tuple)):
        return type(value)(
            _to_device(item, device, non_blocking) for item in value
        )
    return value


# -- load_batch -----------------------------------------------------------------


@dataclass
class LoadBatchConfig:
    """`key` stores the whole batch under one key (default `batch`);
    `fields` names its parts instead. Setting both is refused: one of them
    would be ignored."""

    key: str | None = None
    fields: Any = None
    non_blocking: bool = False

    def __post_init__(self):
        if self.fields is None:
            self.key = _key("batch" if self.key is None else self.key, "key")
        elif self.key is not None:
            raise ValueError(
                "key names the whole batch and fields name its parts; set "
                f"one of them, not both (key={self.key!r}, "
                f"fields={self.fields!r})"
            )
        if self.fields is None:
            pass
        elif isinstance(self.fields, Mapping):
            if not self.fields:
                raise ValueError("fields must not be empty")
            self.fields = {
                _key(key, "fields"): batch_key
                for key, batch_key in self.fields.items()
            }
        elif _is_sequence(self.fields):
            if not self.fields:
                raise ValueError("fields must not be empty")
            self.fields = _unique(
                [_key(key, "fields") for key in self.fields], "fields",
            )
        else:
            raise ValueError(
                "fields must be a list of keys (for a tuple batch) or a "
                "mapping of key to batch field (for a dict batch); got "
                f"{self.fields!r}"
            )
        if not isinstance(self.non_blocking, bool):
            raise ValueError(
                f"non_blocking must be a boolean; got {self.non_blocking!r}"
            )


# data_manager exists per session type, so this is registered per type too.
@requires_resource("data_manager")
@step("load_batch", session_type=TRAINING_SESSION_TYPE)
@step("load_batch", session_type=ANALYSIS_SESSION_TYPE)
class LoadBatch(Step):
    """Take the next batch, move it to the device, and name its parts.

    With `fields` a list, a tuple or list batch is unpacked into one key per
    element; with `fields` a mapping, a dict batch is picked apart by field.
    Otherwise the whole batch is stored under `key`.
    """

    config_schema = LoadBatchConfig

    @override
    def context_writes(self) -> dict[str, str]:
        fields = self._cfg.fields
        keys = (self._cfg.key,) if fields is None else tuple(fields)
        return {key: key for key in keys}

    @override
    def run(self, session: Session) -> Any:
        batch = _to_device(
            next(self.get_dependency("data_manager").data_iter),
            session.device,
            self._cfg.non_blocking,
        )
        fields = self._cfg.fields
        if fields is None:
            return batch
        if isinstance(fields, Mapping):
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"{self.name}.fields picks fields of a dict batch, but "
                    f"the batch is a {type(batch).__name__}"
                )
            values = {}
            for key, batch_key in fields.items():
                try:
                    values[key] = batch[batch_key]
                except KeyError as error:
                    raise KeyError(
                        f"{self.name}: the batch has no field {batch_key!r}; "
                        f"it has {sorted(map(str, batch))}"
                    ) from error
            return _returned(values)
        if not isinstance(batch, (list, tuple)) or len(batch) != len(fields):
            size = len(batch) if isinstance(batch, (list, tuple)) else None
            if isinstance(batch, Mapping):
                hint = (
                    " A dict batch is picked apart with a mapping of key to "
                    "field, e.g. fields: {inputs: <field>}."
                )
            elif size is None:
                hint = (
                    " fields unpacks a tuple or list batch; to store this "
                    f"batch whole, set key: {fields[0]} instead."
                )
            else:
                hint = ""
            raise ValueError(
                f"{self.name}.fields names {len(fields)} parts, but the "
                f"batch is a {type(batch).__name__}"
                + (f" of {size}" if size is not None else "")
                + "." + hint
            )
        return _returned(dict(zip(fields, batch)))


# -- calling a model or a function --------------------------------------------------


@dataclass(kw_only=True)
class CallConfig:
    """How a call takes its inputs from, and puts its result into, the
    iteration context."""

    outputs: Any
    args: Any = ()
    kwargs: Any = field(default_factory=dict)
    constants: Any = field(default_factory=dict)
    no_grad: bool = False

    def __post_init__(self):
        if not _is_sequence(self.args):
            raise ValueError(
                f"args must be a list of keys (or lists of keys); got {self.args!r}"
            )
        self.args = tuple(
            tuple(_key(key, f"args[{index}]") for key in arg)
            if _is_sequence(arg)
            else _key(arg, f"args[{index}]")
            for index, arg in enumerate(self.args)
        )
        self.kwargs = {
            parameter: _key(key, f"kwargs.{parameter}")
            for parameter, key in _parameter_names(self.kwargs, "kwargs").items()
        }
        self.constants = _parameter_names(self.constants, "constants")
        both = sorted(set(self.kwargs) & set(self.constants))
        if both:
            raise ValueError(
                f"parameters {both} are given both by kwargs and constants"
            )
        if isinstance(self.outputs, str):
            self.outputs = _key(self.outputs, "outputs")
        elif isinstance(self.outputs, Mapping):
            if not self.outputs:
                raise ValueError("outputs must not be empty")
            self.outputs = {
                _key(key, "outputs"): result_field
                for key, result_field in self.outputs.items()
            }
        elif _is_sequence(self.outputs) and self.outputs:
            self.outputs = _unique(
                [_key(key, "outputs") for key in self.outputs], "outputs",
            )
        else:
            raise ValueError(
                "outputs must be a key, a list of keys (to unpack a tuple "
                "result) or a mapping of key to result field; got "
                f"{self.outputs!r}"
            )
        if not isinstance(self.no_grad, bool):
            raise ValueError(f"no_grad must be a boolean; got {self.no_grad!r}")


class _CallStep(Step):
    """Calls something on keys of the iteration context and stores the
    result under other keys; its reads and writes come from that config."""

    # Keys need not be identifiers, so reads arrive by key through **inputs;
    # `self` and `session` are positional-only in `run`, so any key -- even
    # one named `session` -- is free to be one of them.
    @override
    def context_reads(self) -> dict[str, str]:
        keys: list[str] = []
        for arg in self._cfg.args:
            keys.extend([arg] if isinstance(arg, str) else arg)
        keys.extend(self._cfg.kwargs.values())
        return {key: key for key in dict.fromkeys(keys)}

    @override
    def context_writes(self) -> dict[str, str]:
        outputs = self._cfg.outputs
        keys = (outputs,) if isinstance(outputs, str) else tuple(outputs)
        return {key: key for key in keys}

    def _call(self, target, inputs: Mapping[str, Any]) -> Any:
        args = [
            inputs[arg] if isinstance(arg, str) else [inputs[key] for key in arg]
            for arg in self._cfg.args
        ]
        kwargs = {
            parameter: inputs[key]
            for parameter, key in self._cfg.kwargs.items()
        }
        kwargs.update(self._cfg.constants)
        grad_mode = torch.no_grad() if self._cfg.no_grad else contextlib.nullcontext()
        with grad_mode:
            result = target(*args, **kwargs)
        return self._outputs(result)

    def _outputs(self, result) -> Any:
        """The call's result as this step returns it, picked or unpacked
        into the configured `outputs`."""
        outputs = self._cfg.outputs
        if isinstance(outputs, str):
            return result
        values = {}
        if isinstance(outputs, Mapping):
            for key, result_field in outputs.items():
                if isinstance(result, Mapping):
                    if result_field not in result:
                        raise KeyError(
                            f"{self.name}.outputs picks {result_field!r}, which "
                            f"the result does not have; it has "
                            f"{sorted(map(str, result))}"
                        )
                    values[key] = result[result_field]
                elif isinstance(result_field, str) and hasattr(result, result_field):
                    values[key] = getattr(result, result_field)
                else:
                    raise KeyError(
                        f"{self.name}.outputs picks {result_field!r}, which a "
                        f"{type(result).__name__} result does not have"
                    )
            return _returned(values)
        if not isinstance(result, (list, tuple)) or len(result) != len(outputs):
            raise ValueError(
                f"{self.name}.outputs unpacks {len(outputs)} values, but "
                f"the call returned a {type(result).__name__}"
                + (
                    f" of {len(result)}"
                    if isinstance(result, (list, tuple)) else ""
                )
            )
        return _returned(dict(zip(outputs, result)))


def _shares_parameters(module: nn.Module, other: nn.Module) -> bool:
    theirs = {id(parameter) for parameter in other.parameters()}
    return any(id(parameter) in theirs for parameter in module.parameters())


def _bound_method(model, method: str, step_name: str):
    target = getattr(model, method, None)
    if not callable(target):
        raise AttributeError(
            f"{step_name}.method is {method!r}, which "
            f"{type(model).__name__} does not have"
        )
    return target


# -- forward ------------------------------------------------------------------------


@dataclass(kw_only=True)
class ForwardConfig(CallConfig):
    method: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.method is not None and (
                not isinstance(self.method, str) or not self.method
        ):
            raise ValueError(f"method must be a method name; got {self.method!r}")


@requires_resource("ddp")
@requires_resource("model")
@step("forward", session_type=TRAINING_SESSION_TYPE)
class Forward(_CallStep):
    """Call a model on keys of the iteration context.

    The model is the `model` resource, or another one for this instance
    through its bindings (`forward#teacher: {model: teacher}`). The model DDP
    wraps is always called through the wrapper, so gradients are
    synchronised; `method` calls one of its methods directly instead, which
    bypasses DDP and suits paths without gradients.
    """

    config_schema = ForwardConfig

    def __init__(self, config=None):
        super().__init__(config)
        self._route_checked = False

    @override
    def run(self, session: Session, /, **inputs: Any) -> Any:
        return self._call(self._target(), inputs)

    def _target(self):
        model = self.get_dependency("model")
        wrapped = self.get_dependency("ddp").wrapped_model
        if not self._route_checked:
            self._check_route(model, getattr(wrapped, "module", None))
            self._route_checked = True
        if self._cfg.method is not None:
            return _bound_method(model, self._cfg.method, self.name)
        if getattr(wrapped, "module", None) is model:
            return wrapped
        if not callable(model):
            raise TypeError(f"{self.name}: {type(model).__name__} is not callable")
        return model

    def _check_route(self, model, wrapped_module) -> None:
        """Refuse a call that would run the parameters DDP wraps outside the
        wrapper with gradients on.

        DDP averages a gradient across ranks only for a forward pass it ran.
        Called directly -- by `method`, or through a module bound here that
        shares parameters with the wrapped one -- those gradients stay local
        to each rank, so every rank trains its own copy, and nothing fails.
        Refused on any world size: a configuration valid on one rank has to
        be valid on eight. A bound object that is not a module but reaches
        into the wrapped model itself cannot be seen here.
        """
        if self._cfg.no_grad or wrapped_module is None:
            return
        method = self._cfg.method
        if model is wrapped_module:
            if method is None:
                return
            called = f"{type(model).__name__}.{method}"
        elif isinstance(model, nn.Module) and _shares_parameters(
                model, wrapped_module,
        ):
            called = type(model).__name__ + (f".{method}" if method else "")
        else:
            return
        raise RuntimeError(
            f"{self.name} calls {called} directly, not through the DDP "
            "wrapper, with gradients on: DDP would not average those "
            "gradients across ranks, so each rank would train its own copy. "
            "Call it through the wrapper (drop `method`, or bind the model "
            "DDP wraps), or set no_grad: true for a path without gradients."
        )


@dataclass(kw_only=True)
class AnalysisForwardConfig(ForwardConfig):
    no_grad: bool = True


@requires_resource("trained_model")
@step("forward", session_type=ANALYSIS_SESSION_TYPE)
class AnalysisForward(_CallStep):
    """Call the trained model on keys of the iteration context.

    Without gradients unless `no_grad: false` asks for them (e.g. for
    gradient-based attributions).
    """

    config_schema = AnalysisForwardConfig

    @override
    def run(self, session: Session, /, **inputs: Any) -> Any:
        model = self.get_dependency("trained_model").model
        target = (
            model
            if self._cfg.method is None
            else _bound_method(model, self._cfg.method, self.name)
        )
        return self._call(target, inputs)


# -- compute ------------------------------------------------------------------------


_FUNCTION_NAMESPACES = (
    ("training_framework.functions", builtin_functions),
    ("torch.nn", nn),
    ("torch.nn.functional", nn.functional),
    ("torch", torch),
)


def _resolve_function(name: Any, init: Mapping[str, Any]):
    if not isinstance(name, str) or not name:
        raise ValueError(f"function must be a name or dotted path; got {name!r}")
    if "." in name:
        resolved = import_object(
            name, "compute.function", example="torch.nn.functional.mse_loss",
        )
    else:
        for _, namespace in _FUNCTION_NAMESPACES:
            resolved = getattr(namespace, name, None)
            if resolved is not None:
                break
        else:
            searched = ", ".join(label for label, _ in _FUNCTION_NAMESPACES)
            raise ValueError(
                f"compute.function {name!r} is not in {searched}; give a "
                "dotted path to your own function or class"
            )
    # A module (a loss, a small network) is built once and then called; so
    # is any class given `init`, even an empty one. Any other class -- `int`,
    # a result type -- is called directly, like a function. `init` absent
    # (None) and `init: {}` therefore mean different things.
    if isinstance(resolved, type) and (
            issubclass(resolved, nn.Module) or init is not None
    ):
        try:
            resolved = resolved(**(init or {}))
        except TypeError as error:
            raise ValueError(
                f"compute.init does not fit {name!r}: {error}"
            ) from error
    elif init is not None:
        raise ValueError(
            f"compute.init is for a class, but {name!r} is not one"
        )
    if not callable(resolved):
        raise TypeError(f"compute.function {name!r} is not callable")
    return resolved


@dataclass(kw_only=True)
class ComputeConfig(CallConfig):
    function: str
    init: Any = None
    """Constructor arguments; present (even `{}`) means "construct once"."""

    def __post_init__(self):
        super().__post_init__()
        if self.init is not None:
            self.init = _parameter_names(self.init, "init")


@step("compute")
class Compute(_CallStep):
    """Call a function, or an instance of a class, on context keys.

    `function` is a name from `training_framework.functions`, `torch.nn`,
    `torch.nn.functional` or `torch` (searched in that order), or a dotted
    path. An `nn.Module` class, or any class given `init` (even `{}`), is
    constructed once and the instance is called (a module is moved to the
    session's device); any other class is called directly, like a function.

    Meant for stateless callables. The instance persists across iterations
    but is not checkpointed, and a module's parameters are not trained: keep
    learnable weights in the model or a `ModuleResource`, and other state in
    a `StatefulStep`.
    """

    config_schema = ComputeConfig

    def __init__(self, config=None):
        super().__init__(config)
        self._function = _resolve_function(self._cfg.function, self._cfg.init)
        self._device = None

    @override
    def run(self, session: Session, /, **inputs: Any) -> Any:
        if isinstance(self._function, nn.Module) and self._device != session.device:
            self._function.to(session.device)
            self._device = session.device
        return self._call(self._function, inputs)


__all__ = [
    "AnalysisForward",
    "Compute",
    "Forward",
    "LoadBatch",
]
