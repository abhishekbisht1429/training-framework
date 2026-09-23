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
    key: str = "batch"
    fields: Any = None
    non_blocking: bool = False

    def __post_init__(self):
        self.key = _key(self.key, "key")
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
    def context_writes(self) -> tuple[str, ...]:
        fields = self._cfg.fields
        if fields is None:
            return (self._cfg.key,)
        return tuple(fields)

    @override
    def run(self, session: Session) -> None:
        batch = _to_device(
            next(self.get_dependency("data_manager").data_iter),
            session.device,
            self._cfg.non_blocking,
        )
        context = session.iteration_context
        fields = self._cfg.fields
        if fields is None:
            context[self._cfg.key] = batch
        elif isinstance(fields, Mapping):
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"{self.name}.fields picks fields of a dict batch, but "
                    f"the batch is a {type(batch).__name__}"
                )
            for key, batch_key in fields.items():
                try:
                    context[key] = batch[batch_key]
                except KeyError as error:
                    raise KeyError(
                        f"{self.name}: the batch has no field {batch_key!r}; "
                        f"it has {sorted(map(str, batch))}"
                    ) from error
        else:
            if not isinstance(batch, (list, tuple)) or len(batch) != len(fields):
                size = len(batch) if isinstance(batch, (list, tuple)) else None
                raise ValueError(
                    f"{self.name}.fields names {len(fields)} parts, but the "
                    f"batch is a {type(batch).__name__}"
                    + (f" of {size}" if size is not None else "")
                )
            for key, value in zip(fields, batch):
                context[key] = value


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

    @override
    def context_reads(self) -> tuple[str, ...]:
        keys: list[str] = []
        for arg in self._cfg.args:
            keys.extend([arg] if isinstance(arg, str) else arg)
        keys.extend(self._cfg.kwargs.values())
        return tuple(dict.fromkeys(keys))

    @override
    def context_writes(self) -> tuple[str, ...]:
        outputs = self._cfg.outputs
        return (outputs,) if isinstance(outputs, str) else tuple(outputs)

    def _call(self, session: Session, target) -> None:
        context = session.iteration_context
        args = [
            context[arg] if isinstance(arg, str) else [context[key] for key in arg]
            for arg in self._cfg.args
        ]
        kwargs = {
            parameter: context[key]
            for parameter, key in self._cfg.kwargs.items()
        }
        kwargs.update(self._cfg.constants)
        grad_mode = torch.no_grad() if self._cfg.no_grad else contextlib.nullcontext()
        with grad_mode:
            result = target(*args, **kwargs)
        self._store(context, result)

    def _store(self, context, result) -> None:
        outputs = self._cfg.outputs
        if isinstance(outputs, str):
            context[outputs] = result
        elif isinstance(outputs, Mapping):
            for key, result_field in outputs.items():
                if isinstance(result, Mapping):
                    if result_field not in result:
                        raise KeyError(
                            f"{self.name}.outputs picks {result_field!r}, which "
                            f"the result does not have; it has "
                            f"{sorted(map(str, result))}"
                        )
                    context[key] = result[result_field]
                elif isinstance(result_field, str) and hasattr(result, result_field):
                    context[key] = getattr(result, result_field)
                else:
                    raise KeyError(
                        f"{self.name}.outputs picks {result_field!r}, which a "
                        f"{type(result).__name__} result does not have"
                    )
        else:
            if not isinstance(result, (list, tuple)) or len(result) != len(outputs):
                raise ValueError(
                    f"{self.name}.outputs unpacks {len(outputs)} values, but "
                    f"the call returned a {type(result).__name__}"
                    + (
                        f" of {len(result)}"
                        if isinstance(result, (list, tuple)) else ""
                    )
                )
            for key, value in zip(outputs, result):
                context[key] = value


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

    @override
    def run(self, session: Session) -> None:
        self._call(session, self._target())

    def _target(self):
        model = self.get_dependency("model")
        if self._cfg.method is not None:
            return _bound_method(model, self._cfg.method, self.name)
        wrapped = self.get_dependency("ddp").wrapped_model
        if getattr(wrapped, "module", None) is model:
            return wrapped
        if not callable(model):
            raise TypeError(f"{self.name}: {type(model).__name__} is not callable")
        return model


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
    def run(self, session: Session) -> None:
        model = self.get_dependency("trained_model").model
        target = (
            model
            if self._cfg.method is None
            else _bound_method(model, self._cfg.method, self.name)
        )
        self._call(session, target)


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
    """

    config_schema = ComputeConfig

    def __init__(self, config=None):
        super().__init__(config)
        self._function = _resolve_function(self._cfg.function, self._cfg.init)
        self._device = None

    @override
    def run(self, session: Session) -> None:
        if isinstance(self._function, nn.Module) and self._device != session.device:
            self._function.to(session.device)
            self._device = session.device
        self._call(session, self._function)


__all__ = [
    "AnalysisForward",
    "Compute",
    "Forward",
    "LoadBatch",
]
