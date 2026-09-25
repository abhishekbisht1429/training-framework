"""Optimization as a resource and the steps that drive it.

The `optimizer` resource owns the torch optimizer, its learning-rate
schedule and, for fp16, the gradient scaler: it builds them in `setup`,
checkpoints their state, and keeps the precision and accumulation settings.
Per-iteration work is done by steps, ordered by the steps they require:

    <loss step> -> backward -> freeze_gradients -> clip_gradients -> optimizer_step

`forward_context` is the one hook: autocast and DDP's `no_sync` have to be
entered before the forward pass, which a step cannot guarantee to precede.
Configuring `optimizer` activates `optimizer_step`, which brings in the rest
of the chain; `backward` reads `iteration_context["loss"]` (its `loss_key`)
as its `loss` argument, so it runs after whichever step declares writing it.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, override

import torch
from torch import empty, nn, optim

from training_framework.components import (
    ExtendableComponent,
    LifecycleHook,
    StatefulResource,
    Step,
    activates,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
)

if TYPE_CHECKING:
    from training_framework.session import Session


_LEGACY_CONFIG_KEYS = frozenset({
    "learning_rate",
    "weight_decay",
    "warmup_iters",
})
_CONFIG_KEYS = frozenset({
    "optimizer",
    "lr_scheduler",
    "param_groups",
    "precision",
    "accumulate_steps",
})
_SPEC_KEYS = frozenset({"name", "kwargs"})
_SCHEDULER_KEYS = frozenset({"stages", "milestones", "metric_key"})
_PARAM_GROUP_KEYS = frozenset({"match", "kwargs"})
_RUNTIME_PLACEHOLDERS = frozenset({
    "$max_iterations",
    "$stage_iterations",
})
_PRECISIONS: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

def _unknown_keys(config: Mapping, allowed: frozenset[str]) -> list[str]:
    return sorted(str(key) for key in set(config) - allowed)


def _supports_extension_path(path: tuple[str, ...]) -> bool:
    if path[:2] == ("optimizer", "kwargs") and len(path) >= 3:
        return True
    return bool(path) and path[0] == "lr_scheduler"


def _rebase_scheduler_lrs(
        state: Mapping,
        new_lr: float,
        keep: frozenset[int] = frozenset(),
) -> None:
    """Rescale a serialized scheduler state so its current lr becomes
    ``new_lr``, preserving the schedule's relative progress.

    Used when an extension changes the optimizer's ``lr`` while the
    lr_scheduler configuration is unchanged. Overwriting ``base_lrs`` with
    ``new_lr`` directly would ignore whatever multiplier the schedule is
    currently applying (mid-warmup, a decay factor other than 1, ...), so
    each base lr is instead scaled by the ratio between ``new_lr`` and the
    lr the schedule last computed (``_last_lr``). ``_last_lr`` itself is set
    to ``new_lr`` so the override is visible immediately.

    For a SequentialLR, only the currently active stage (per ``_milestones``
    and ``last_epoch``) is rebased this way. A stage that has not started
    yet carries no meaningful "current lr" of its own to preserve — its
    ``_last_lr`` is a leftover from when every stage was independently
    constructed, not a value it ever actually ran at — so it is left alone
    and uses its own originally configured base once it activates.

    Groups whose index is in ``keep`` -- those whose ``param_groups`` entry
    sets its own ``lr`` -- do not follow the global lr, so their entries
    are left as they are and their schedule carries on unchanged.

    A no-op for scheduler kinds with no ``base_lrs`` (e.g.
    ReduceLROnPlateau, which reads the optimizer's current lr directly).
    """
    schedulers = state.get("_schedulers")
    if schedulers:
        milestones = state.get("_milestones", [])
        last_epoch = state.get("last_epoch", 0)
        active_index = bisect_right(milestones, last_epoch)
        _rebase_scheduler_lrs(schedulers[active_index], new_lr, keep)
        return
    if "base_lrs" not in state:
        return
    base_lrs = state["base_lrs"]
    last_lr = state.get("_last_lr", base_lrs)
    state["base_lrs"] = [
        base if index in keep
        else base * (new_lr / current) if current else new_lr
        for index, (base, current) in enumerate(zip(base_lrs, last_lr))
    ]
    state["_last_lr"] = [
        current if index in keep else new_lr
        for index, current in enumerate(last_lr)
    ]


def _require_mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _normalize_spec(value: Any, path: str) -> dict[str, Any]:
    config = _require_mapping(value, path)
    unknown = _unknown_keys(config, _SPEC_KEYS)
    if unknown:
        raise ValueError(f"Unknown {path} fields: {', '.join(unknown)}")
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}.name must be a non-empty string")
    kwargs = _require_mapping(config.get("kwargs", {}), f"{path}.kwargs")
    return {"name": name.strip(), "kwargs": deepcopy(dict(kwargs))}


def _resolve_class(namespace, name: str, expected_base: type, kind: str) -> type:
    candidate = getattr(namespace, name, None)
    if candidate is None:
        raise ValueError(f"Unknown {kind} class: {name}")
    if not isinstance(candidate, type) or not issubclass(candidate, expected_base):
        raise ValueError(
            f"{kind} class {name!r} must extend {expected_base.__name__}"
        )
    return candidate


def _resolve_placeholders(
        value: Any,
        *,
        max_iterations: int,
        stage_iterations: int,
) -> Any:
    if isinstance(value, str):
        if value == "$max_iterations":
            return max_iterations
        if value == "$stage_iterations":
            return stage_iterations
        if value.startswith("$"):
            raise ValueError(
                f"Unknown scheduler runtime placeholder {value!r}; expected "
                f"one of {', '.join(sorted(_RUNTIME_PLACEHOLDERS))}"
            )
        return value
    if isinstance(value, Mapping):
        return {
            key: _resolve_placeholders(
                item,
                max_iterations=max_iterations,
                stage_iterations=stage_iterations,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_placeholders(
                item,
                max_iterations=max_iterations,
                stage_iterations=stage_iterations,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_placeholders(
                item,
                max_iterations=max_iterations,
                stage_iterations=stage_iterations,
            )
            for item in value
        )
    return deepcopy(value)



def _patterns(value: Any, path: str) -> list[str]:
    """Return a list of parameter-name glob patterns, one string or several."""
    if isinstance(value, str):
        value = [value]
    if (
            not isinstance(value, Sequence)
            or isinstance(value, (bytes, Mapping))
            or not value
            or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(
            f"{path} must be a non-empty list of parameter name patterns"
        )
    return list(value)


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _check_patterns_match(
        patterns: Sequence[str],
        names: Sequence[str],
        path: str,
) -> None:
    """Reject a pattern that selects no parameter: it is almost always a typo,
    and a rule that silently applies to nothing gives a run that works and is
    quietly wrong."""
    unmatched = [
        pattern for pattern in patterns
        if not any(fnmatchcase(name, pattern) for name in names)
    ]
    if unmatched:
        preview = ", ".join(names[:8]) + (", ..." if len(names) > 8 else "")
        raise ValueError(
            f"{path} patterns {unmatched} match no parameter. Parameter "
            f"names look like: {preview}"
        )


def _unwrapped(wrapped_model: nn.Module) -> nn.Module:
    """The model DDP wraps, whose parameter names patterns are written for."""
    if isinstance(wrapped_model, nn.parallel.DistributedDataParallel):
        return wrapped_model.module
    return wrapped_model


@dataclass(frozen=True)
class _OptimizerSettings:
    optimizer_spec: dict[str, Any]
    scheduler_config: dict[str, Any] | None
    param_groups: list[dict[str, Any]]
    precision: str
    accumulate_steps: int


@activates("optimizer_step")
@requires_resource("ddp")
@resource("optimizer", session_type="training")
class OptimizerResource(StatefulResource, ExtendableComponent):
    """The optimizer, its schedule and gradient scaler, and how they step.

    Built in `setup` from the parameters of the DDP-wrapped model and dropped
    in `teardown`, keeping their state across both, as a checkpoint does.
    The steps of the optimization chain drive it through the methods below;
    `forward_context` opens each iteration with `begin_iteration`.
    """

    def __init__(self, config):
        self._settings = self._normalize_config(config)
        self._optimizer = None
        self._lr_scheduler = None
        self._grad_scaler = None
        self._module = None
        self._restored_state = None
        self._max_iterations = None
        # How many optimizer steps the schedule was built for; see step.
        self._schedule_steps = None
        self._device_type = None
        # Per iteration; see begin_iteration.
        self._boundary = True
        self._autocast = None
        self._no_sync = None
        self._processed: set[int] = set()
        self._grad_norm: float | None = None
        # The iteration begun, how many micro-batches its accumulation group
        # has and where it starts, and the last iteration whose backward
        # ran -- which tells a retried iteration from a fresh one.
        self._iteration: int | None = None
        self._group_size = 1
        self._group_start = 1
        self._backward_iteration: int | None = None

    # -- configuration ------------------------------------------------------

    @staticmethod
    def _normalize_config(config) -> _OptimizerSettings:
        config = _require_mapping(config, "optimizer config")
        legacy = sorted(_LEGACY_CONFIG_KEYS & set(config))
        if legacy:
            raise ValueError(
                "Legacy optimizer fields are no longer supported: "
                f"{', '.join(legacy)}. Use optimizer.optimizer.name and "
                "optimizer.optimizer.kwargs; configure scheduling under "
                "optimizer.lr_scheduler."
            )
        unknown = _unknown_keys(config, _CONFIG_KEYS)
        if unknown:
            raise ValueError(
                "Unknown optimizer config fields: " + ", ".join(unknown)
            )
        if "optimizer" not in config:
            raise ValueError("optimizer.optimizer is required")

        optimizer_spec = _normalize_spec(
            config["optimizer"], "optimizer.optimizer"
        )
        _resolve_class(
            optim, optimizer_spec["name"], optim.Optimizer, "optimizer"
        )
        if "params" in optimizer_spec["kwargs"]:
            raise ValueError(
                "optimizer.optimizer.kwargs must not contain 'params'; model "
                "parameters are supplied by the optimizer resource"
            )

        precision = config.get("precision", "fp32")
        if precision not in _PRECISIONS:
            raise ValueError(
                f"optimizer.precision must be one of {sorted(_PRECISIONS)}; "
                f"got {precision!r}"
            )
        accumulate_steps = config.get("accumulate_steps", 1)
        if (
                isinstance(accumulate_steps, bool)
                or not isinstance(accumulate_steps, int)
                or accumulate_steps <= 0
        ):
            raise ValueError(
                "optimizer.accumulate_steps must be a positive integer; got "
                f"{accumulate_steps!r}"
            )

        return _OptimizerSettings(
            optimizer_spec=optimizer_spec,
            scheduler_config=OptimizerResource._normalize_scheduler(
                config.get("lr_scheduler")
            ),
            param_groups=OptimizerResource._normalize_param_groups(
                config.get("param_groups")
            ),
            precision=precision,
            accumulate_steps=accumulate_steps,
        )

    @staticmethod
    def _normalize_param_groups(value) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(
                value, Sequence,
        ):
            raise TypeError(
                "optimizer.param_groups must be a list of {match, kwargs} "
                "mappings"
            )
        groups = []
        for index, entry in enumerate(value):
            path = f"optimizer.param_groups[{index}]"
            entry = _require_mapping(entry, path)
            unknown = _unknown_keys(entry, _PARAM_GROUP_KEYS)
            if unknown:
                raise ValueError(f"Unknown {path} fields: {', '.join(unknown)}")
            kwargs = _require_mapping(entry.get("kwargs", {}), f"{path}.kwargs")
            if "params" in kwargs:
                raise ValueError(
                    f"{path}.kwargs must not contain 'params'; the group's "
                    "parameters are the ones its patterns match"
                )
            groups.append({
                "match": _patterns(entry.get("match"), f"{path}.match"),
                "kwargs": deepcopy(dict(kwargs)),
            })
        return groups

    @staticmethod
    def _normalize_scheduler(scheduler_value) -> dict[str, Any] | None:
        if scheduler_value is None:
            return None
        scheduler = _require_mapping(
            scheduler_value, "optimizer.lr_scheduler"
        )
        unknown = _unknown_keys(scheduler, _SCHEDULER_KEYS)
        if unknown:
            raise ValueError(
                "Unknown optimizer.lr_scheduler fields: "
                + ", ".join(unknown)
            )

        stages_value = scheduler.get("stages")
        if (
                not isinstance(stages_value, Sequence)
                or isinstance(stages_value, (str, bytes))
                or not stages_value
        ):
            raise ValueError(
                "optimizer.lr_scheduler.stages must be a non-empty list"
            )
        stages = []
        for index, stage_value in enumerate(stages_value):
            stage = _normalize_spec(
                stage_value, f"optimizer.lr_scheduler.stages[{index}]"
            )
            _resolve_class(
                optim.lr_scheduler,
                stage["name"],
                optim.lr_scheduler.LRScheduler,
                "scheduler",
            )
            if "optimizer" in stage["kwargs"]:
                raise ValueError(
                    "Scheduler kwargs must not contain 'optimizer'; it is "
                    "supplied by the optimizer resource"
                )
            stages.append(stage)

        milestones_value = scheduler.get("milestones", [])
        if (
                not isinstance(milestones_value, Sequence)
                or isinstance(milestones_value, (str, bytes))
        ):
            raise TypeError(
                "optimizer.lr_scheduler.milestones must be a list of integers"
            )
        milestones = list(milestones_value)
        if len(milestones) != len(stages) - 1:
            raise ValueError(
                "optimizer.lr_scheduler.milestones must contain exactly one "
                "entry between each pair of stages"
            )
        if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in milestones
        ):
            raise TypeError(
                "optimizer.lr_scheduler.milestones must be a list of integers"
            )
        if any(value <= 0 for value in milestones) or any(
                left >= right
                for left, right in zip(milestones, milestones[1:])
        ):
            raise ValueError(
                "optimizer.lr_scheduler.milestones must be positive and "
                "strictly increasing"
            )

        metric_key = scheduler.get("metric_key")
        if metric_key is not None:
            if not isinstance(metric_key, str) or not metric_key.strip():
                raise ValueError(
                    "optimizer.lr_scheduler.metric_key must be a non-empty "
                    "string"
                )
            if len(stages) != 1:
                raise ValueError(
                    "optimizer.lr_scheduler.metric_key is supported only for "
                    "a single scheduler stage"
                )
            metric_key = metric_key.strip()

        return {
            "stages": stages,
            "milestones": milestones,
            "metric_key": metric_key,
        }

    @property
    def precision(self) -> str:
        return self._settings.precision

    @property
    def accumulate_steps(self) -> int:
        return self._settings.accumulate_steps

    def optimizer_steps(self, max_iterations: int) -> int:
        """How many times the optimizer steps in `max_iterations` iterations.

        The schedule advances once per optimizer step, so this is what
        `$max_iterations` and the milestones are counted in.
        """
        return math.ceil(max_iterations / self._settings.accumulate_steps)

    def _optimizer_steps_left(self, completed_iterations: int) -> int:
        """How many times the optimizer steps after `completed_iterations`:
        on every k-th iteration and on the final one."""
        return (
            self.optimizer_steps(self._max_iterations)
            - completed_iterations // self._settings.accumulate_steps
        )

    # -- construction -------------------------------------------------------

    def _build_optimizer(self, wrapped_model: nn.Module) -> optim.Optimizer:
        spec = self._settings.optimizer_spec
        optimizer_class = _resolve_class(
            optim, spec["name"], optim.Optimizer, "optimizer",
        )
        try:
            return optimizer_class(
                self._parameter_groups(wrapped_model),
                **deepcopy(spec["kwargs"]),
            )
        except TypeError as error:
            raise ValueError(
                f"Invalid kwargs for optimizer {spec['name']!r}: {error}"
            ) from error

    def _parameter_groups(self, wrapped_model: nn.Module):
        configured = self._settings.param_groups
        if not configured:
            # One group of every parameter, in model order: the layout an
            # optimizer state saved without param_groups has.
            return wrapped_model.parameters()

        named = list(self._module.named_parameters())
        names = [name for name, _ in named]
        claimed: set[str] = set()
        groups = []
        for index, group in enumerate(configured):
            _check_patterns_match(
                group["match"], names, f"optimizer.param_groups[{index}].match",
            )
            params = []
            for name, parameter in named:
                if name not in claimed and _matches(name, group["match"]):
                    claimed.add(name)
                    params.append(parameter)
            groups.append({"params": params, **deepcopy(group["kwargs"])})
        rest = [parameter for name, parameter in named if name not in claimed]
        # Always present, even when empty, so group indices -- and a saved
        # state's layout -- depend only on the configuration.
        groups.append({"params": rest})
        return groups

    def _prepare_scheduler(self, optimizer, total_steps):
        """Build the schedule over `total_steps` optimizer steps, which
        `$max_iterations` resolves to."""
        scheduler_config = self._settings.scheduler_config
        if scheduler_config is None:
            return None
        milestones = scheduler_config["milestones"]
        if total_steps <= 0:
            raise ValueError(
                "A configured lr_scheduler requires at least one optimizer "
                "step left in the run (session_config.max_iterations)"
            )
        if milestones and milestones[-1] >= total_steps:
            raise ValueError(
                "optimizer.lr_scheduler milestones must be less than the "
                "number of optimizer steps the schedule runs for "
                "(session_config.max_iterations divided by "
                "optimizer.accumulate_steps, counted from where the schedule "
                "started)"
            )

        boundaries = [0, *milestones, total_steps]
        schedulers = []
        for index, spec in enumerate(scheduler_config["stages"]):
            scheduler_class = _resolve_class(
                optim.lr_scheduler,
                spec["name"],
                optim.lr_scheduler.LRScheduler,
                "scheduler",
            )
            kwargs = _resolve_placeholders(
                spec["kwargs"],
                max_iterations=total_steps,
                stage_iterations=boundaries[index + 1] - boundaries[index],
            )
            try:
                schedulers.append(scheduler_class(optimizer, **kwargs))
            except TypeError as error:
                raise ValueError(
                    f"Invalid kwargs for scheduler {spec['name']!r}: {error}"
                ) from error
        if len(schedulers) == 1:
            return schedulers[0]
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones
        )

    def _check_precision_supported(self) -> None:
        if (
                self._settings.precision == "bf16"
                and self._device_type == "cuda"
                and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError(
                "optimizer.precision is bf16, but this CUDA device does not "
                "support bfloat16. Use fp16 or fp32."
            )

    # -- lifecycle ----------------------------------------------------------

    @override
    def setup(self, session: Session) -> None:
        wrapped_model = self.get_dependency("ddp").wrapped_model
        self._module = _unwrapped(wrapped_model)
        self._max_iterations = session.session_config.max_iterations
        self._device_type = session.device.type
        self._check_precision_supported()
        self._optimizer = self._build_optimizer(wrapped_model)
        if self._settings.precision == "fp16":
            self._grad_scaler = torch.amp.GradScaler(self._device_type)

        restored = self._restored_state
        scheduler_state = (
            restored.get("lr_scheduler_state") if restored is not None else None
        )
        if scheduler_state is None:
            # No prior scheduler state will be layered on afterward (first
            # run, or an extension replaced the scheduler): restore the
            # optimizer's lr first, so a freshly constructed scheduler's
            # own initial-step application (e.g. a LinearLR/ConstantLR
            # warmup factor) uses the correct base -- nothing will
            # overwrite it afterward.
            self._restore_optimizer_state(restored)
            # A schedule starting here -- at the start of the run, or at
            # the extension point for a replacement -- runs over the
            # optimizer steps still to come.
            self._schedule_steps = (
                self._optimizer_steps_left(session.iteration)
                if self._settings.scheduler_config is not None
                else None
            )
            self._lr_scheduler = self._prepare_scheduler(
                self._optimizer, self._schedule_steps
            )
            self._restore_scheduler_state(restored)
        else:
            # A schedule in progress keeps the length it was built for; a
            # longer run does not stretch it (see step). A checkpoint
            # written before that length was saved has none.
            self._schedule_steps = restored.get("schedule_steps")
            # An existing scheduler state is restored afterward via
            # load_state_dict, which only syncs the scheduler's own
            # counters (base_lrs, last_epoch, ...) and never touches
            # optimizer.param_groups. So the optimizer_state restore must
            # run last: it's what puts the correct current lr back after
            # the fresh scheduler's construction-time initial step
            # overwrote it with its own (stale, epoch-0) value.
            self._lr_scheduler = self._prepare_scheduler(
                self._optimizer,
                self._schedule_steps
                if self._schedule_steps is not None
                else self.optimizer_steps(self._max_iterations),
            )
            self._restore_optimizer_state(restored)
            self._restore_scheduler_state(restored)
        self._restore_scaler_state(restored)
        self._restored_state = None
        # Gradients from a previous session, or a partly accumulated group,
        # are never carried into a new one.
        self._optimizer.zero_grad()

    def _restore_optimizer_state(self, restored) -> None:
        if restored is None:
            return
        optimizer_state = restored.get("optimizer_state")
        if optimizer_state is None:
            return
        try:
            self._optimizer.load_state_dict(optimizer_state)
        except ValueError as error:
            raise ValueError(
                "The saved optimizer state does not match the optimizer's "
                "parameter groups; optimizer.param_groups (or the model) "
                f"changed since it was saved: {error}"
            ) from error

    def _restore_scheduler_state(self, restored) -> None:
        if restored is None:
            return
        scheduler_state = restored.get("lr_scheduler_state")
        if scheduler_state is not None:
            if self._lr_scheduler is None:
                raise ValueError(
                    "Cannot restore lr_scheduler_state without a configured "
                    "lr_scheduler"
                )
            self._lr_scheduler.load_state_dict(scheduler_state)

    def _restore_scaler_state(self, restored) -> None:
        if restored is None or self._grad_scaler is None:
            return
        scaler_state = restored.get("grad_scaler_state")
        if scaler_state:
            self._grad_scaler.load_state_dict(scaler_state)

    @override
    def teardown(self, session: Session) -> None:
        self._restored_state = self.get_state()
        self._release()

    @override
    def rollback_setup(self, session: Session) -> None:
        self._release()

    def _release(self) -> None:
        self.close_contexts()
        self._optimizer = None
        self._lr_scheduler = None
        self._schedule_steps = None
        self._grad_scaler = None
        self._module = None

    # -- state --------------------------------------------------------------

    @override
    def get_state(self) -> Any:
        if self._optimizer is None:
            if self._restored_state is not None:
                return deepcopy(self._restored_state)
            return {
                "optimizer_state": None,
                "lr_scheduler_state": None,
                "schedule_steps": None,
                "grad_scaler_state": None,
            }
        return {
            "optimizer_state": self._optimizer.state_dict(),
            "lr_scheduler_state": (
                self._lr_scheduler.state_dict()
                if self._lr_scheduler is not None
                else None
            ),
            "schedule_steps": self._schedule_steps,
            "grad_scaler_state": (
                self._grad_scaler.state_dict()
                if self._grad_scaler is not None
                else None
            ),
        }

    @override
    def set_state(self, state: Any) -> None:
        self._restored_state = deepcopy(state)
        if self._optimizer is not None:
            # Everything already exists here (no fresh scheduler
            # construction involved), so restore order doesn't matter the
            # way it does in setup.
            restored = self._restored_state
            self._restore_optimizer_state(restored)
            self._restore_scheduler_state(restored)
            if restored.get("lr_scheduler_state") is not None:
                self._schedule_steps = restored.get("schedule_steps")
            self._restore_scaler_state(restored)
            self._restored_state = None

    # -- extension ----------------------------------------------------------

    @override
    def apply_extension_config(
            self,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        unsupported = {
            path for path in changed_paths if not _supports_extension_path(path)
        }
        if unsupported:
            names = ", ".join(".".join(path) for path in sorted(unsupported))
            raise ValueError(
                "Optimizer session extension does not allow changes to: "
                + names
            )

        settings = self._normalize_config(config)
        if (
                settings.optimizer_spec["name"]
                != self._settings.optimizer_spec["name"]
        ):
            raise ValueError(
                "Optimizer class cannot change during session extension"
            )
        if self._optimizer is not None:
            raise RuntimeError(
                "Optimizer configuration cannot change while the session "
                "is active"
            )

        optimizer_spec = settings.optimizer_spec
        optimizer_class = _resolve_class(
            optim,
            optimizer_spec["name"],
            optim.Optimizer,
            "optimizer",
        )
        try:
            validation_optimizer = optimizer_class(
                [nn.Parameter(empty(1))],
                **deepcopy(optimizer_spec["kwargs"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid kwargs for optimizer "
                f"{optimizer_spec['name']!r}: {error}"
            ) from error
        # The effective base lr for a clean scheduler restart below: reads
        # back whatever the optimizer actually resolved to, whether that
        # came from an explicit `lr` kwarg or the optimizer class's own
        # PyTorch default (e.g. AdamW's 0.001 when `lr` is omitted).
        effective_lr = validation_optimizer.param_groups[0].get("lr")

        changed_kwarg_keys = {
            path[2] for path in changed_paths
            if path[:2] == ("optimizer", "kwargs")
        }
        optimizer_kwargs = optimizer_spec["kwargs"]
        missing = sorted(changed_kwarg_keys - optimizer_kwargs.keys())
        if missing:
            raise ValueError(
                "Optimizer kwargs cannot be removed during session extension: "
                + ", ".join(missing)
            )

        scheduler_changed = (
            settings.scheduler_config != self._settings.scheduler_config
        )
        group_overrides = [
            group["kwargs"] for group in self._settings.param_groups
        ]

        def own_value(index: int, key: str, default):
            # A configured group that sets `key` itself keeps it: the
            # session-wide kwarg is only the default for the others.
            if index < len(group_overrides) and key in group_overrides[index]:
                return group_overrides[index][key]
            return default

        if self._restored_state is not None:
            optimizer_state = self._restored_state.get("optimizer_state")
            if optimizer_state is not None:
                for index, group in enumerate(
                        optimizer_state.get("param_groups", []),
                ):
                    for key in changed_kwarg_keys:
                        group[key] = deepcopy(
                            own_value(index, key, optimizer_kwargs[key])
                        )

            if scheduler_changed:
                # New schedule shape/class: old scheduler state is not
                # guaranteed compatible, so restart schedule progress from
                # the extension point, over the optimizer steps left.
                self._restored_state["lr_scheduler_state"] = None
                self._restored_state["schedule_steps"] = None
                if (
                        settings.scheduler_config is not None
                        and optimizer_state is not None
                        and effective_lr is not None
                ):
                    # A replacement (or newly added) scheduler must start
                    # its own clean schedule from the effective base lr,
                    # not wherever the old scheduler's progress (e.g. a
                    # cosine decay's midpoint) left the optimizer. This
                    # matters even when `lr` itself wasn't part of this
                    # extension -- including when it was never configured
                    # at all and the optimizer is using its own PyTorch
                    # default: leaving group['lr'] at the old scheduler's
                    # current value would corrupt LinearLR/ConstantLR-style
                    # schedulers, which scale *current* group['lr'] (not
                    # base_lrs/initial_lr) at construction time. Also sync
                    # `initial_lr`, the analogous stale value schedulers
                    # that key off base_lrs (e.g. CosineAnnealingWarmRestarts)
                    # would otherwise fall back to via LRScheduler.__init__'s
                    # setdefault.
                    for index, group in enumerate(
                            optimizer_state.get("param_groups", []),
                    ):
                        base_lr = own_value(index, "lr", effective_lr)
                        group["lr"] = deepcopy(base_lr)
                        if "initial_lr" in group:
                            group["initial_lr"] = deepcopy(base_lr)
            elif "lr" in changed_kwarg_keys:
                scheduler_state = self._restored_state.get("lr_scheduler_state")
                if scheduler_state is not None:
                    own_lr = frozenset(
                        index for index, overrides in enumerate(group_overrides)
                        if "lr" in overrides
                    )
                    _rebase_scheduler_lrs(
                        scheduler_state, optimizer_kwargs["lr"], keep=own_lr,
                    )

        self._settings = settings

    # -- the iteration, as the chain drives it ------------------------------

    def begin_iteration(self, iteration: int) -> None:
        """Open an iteration: decide whether it steps, and enter the forward
        contexts (DDP `no_sync` while accumulating, autocast for bf16/fp16).

        Anything a previous iteration left open -- one that raised before
        `backward` -- is closed first.
        """
        self.close_contexts()
        self._processed = set()
        accumulate_steps = self._settings.accumulate_steps
        self._boundary = (
            accumulate_steps == 1
            or iteration % accumulate_steps == 0
            or iteration >= self._max_iterations
        )
        # The group this iteration belongs to. Only the last group of a run
        # can be short; its losses are averaged over its own size.
        self._group_start = (
            (iteration - 1) // accumulate_steps * accumulate_steps + 1
        )
        group_end = min(
            self._group_start + accumulate_steps - 1, self._max_iterations,
        )
        self._group_size = max(1, group_end - self._group_start + 1)
        if self._backward_iteration == iteration:
            self._discard_failed_attempt(iteration)
        self._iteration = iteration
        if not self._boundary:
            no_sync = getattr(
                self.get_dependency("ddp").wrapped_model, "no_sync", None,
            )
            if no_sync is not None:
                context = no_sync()
                context.__enter__()
                self._no_sync = context
        dtype = _PRECISIONS[self._settings.precision]
        if dtype is not None:
            context = torch.autocast(self._device_type, dtype=dtype)
            context.__enter__()
            self._autocast = context

    def _discard_failed_attempt(self, iteration: int) -> None:
        """Undo what a failed attempt at `iteration` left behind.

        The runtime rolls the counter back when an iteration raises, so the
        same iteration can be run again. If the failed attempt got past
        `backward`, its gradients are still there, and at a boundary fp16
        gradients were already unscaled. At the start of a group only that
        attempt contributed, so they are cleared and the scaler gets fresh
        per-step bookkeeping (rebuilt from its own state, keeping its scale).
        Mid-group they cannot be told apart from the micro-batches before it.
        """
        if iteration != self._group_start:
            raise RuntimeError(
                f"Iteration {iteration} failed after its backward pass, in "
                "the middle of a gradient accumulation group: its gradients "
                "cannot be separated from those of the group's earlier "
                "iterations, so it cannot be run again. Resume from the last "
                "checkpoint instead."
            )
        self._optimizer.zero_grad()
        if self._grad_scaler is not None:
            fresh = torch.amp.GradScaler(self._device_type)
            fresh.load_state_dict(self._grad_scaler.state_dict())
            self._grad_scaler = fresh
        self._backward_iteration = None

    @property
    def is_boundary(self) -> bool:
        """Whether this iteration ends an accumulation group and steps."""
        return self._boundary

    def exit_autocast(self) -> None:
        context, self._autocast = self._autocast, None
        if context is not None:
            context.__exit__(None, None, None)

    def exit_no_sync(self) -> None:
        context, self._no_sync = self._no_sync, None
        if context is not None:
            context.__exit__(None, None, None)

    def close_contexts(self) -> None:
        self.exit_autocast()
        self.exit_no_sync()

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate `loss`, scaled for accumulation and fp16.

        Runs outside autocast. At a boundary the fp16 gradients are unscaled
        right away, so every later step sees true gradients.
        """
        self.exit_autocast()
        # A norm is measured only when this iteration steps; until then
        # there is none for it.
        self._grad_norm = None
        try:
            group_size = self._group_size
            scaled = loss / group_size if group_size > 1 else loss
            if self._grad_scaler is not None:
                scaled = self._grad_scaler.scale(scaled)
            self._backward_iteration = self._iteration
            scaled.backward()
        finally:
            self.exit_no_sync()
        if self._boundary and self._grad_scaler is not None:
            self._grad_scaler.unscale_(self._optimizer)

    def parameter_names(self) -> list[str]:
        """Every parameter name of the model, as patterns match them."""
        return [name for name, _ in self._module.named_parameters()]

    def named_gradients(self) -> list[tuple[str, nn.Parameter]]:
        """The model's parameters that have a gradient, by name."""
        return [
            (name, parameter)
            for name, parameter in self._module.named_parameters()
            if parameter.grad is not None
        ]

    def mark_processed(self, processor: Step) -> None:
        self._processed.add(id(processor))

    def check_processed(self, processors: Iterable[Step]) -> None:
        """Refuse to step before every one of `processors` has run.

        A processor ordered after `optimizer_step` -- a custom stage whose
        binding was forgotten -- would edit gradients the step has already
        applied. Caught here, before the first such step is taken.
        `processors` are the session's stages bound to this optimizer, taken
        from the session when the step runs, so a stage replaced or removed
        since is never waited for.
        """
        late = [
            getattr(processor, "name", type(processor).__name__)
            for processor in processors
            if id(processor) not in self._processed
        ]
        if late:
            raise RuntimeError(
                f"Gradient processors {late} had not run when optimizer_step "
                "was about to step. Put each in the chain by binding "
                "optimizer_step to it, e.g. component_bindings: "
                f"{{optimizer_step: {{clip_gradients: {late[0]}}}}}, and "
                "have it require the stage it follows."
            )

    @property
    def metric_key(self) -> str | None:
        """The iteration_context key a metric-driven schedule steps on."""
        scheduler_config = self._settings.scheduler_config
        if scheduler_config is None:
            return None
        return scheduler_config["metric_key"]

    def scheduler_metric(self, values: Mapping) -> Any:
        """The value a metric-driven schedule steps on, looked up in
        `values` by `metric_key`; None if no metric is configured."""
        metric_key = self.metric_key
        if metric_key is None:
            return None
        try:
            return values[metric_key]
        except KeyError as error:
            raise KeyError(
                f"Configured lr_scheduler metric {metric_key!r} is missing "
                f"from the values given; they have {sorted(map(str, values))}"
            ) from error

    def step(self, metric: Any = None) -> None:
        """Apply the gradients, advance the schedule, and clear them.

        Under fp16 the scaler skips a step whose gradients overflowed; the
        schedule then does not advance either, since no step was taken.
        """
        skipped = False
        if self._grad_scaler is not None:
            scale = self._grad_scaler.get_scale()
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
            # `update` lowers the scale only when it found non-finite
            # gradients, which is exactly when `step` skipped.
            skipped = self._grad_scaler.get_scale() < scale
        else:
            self._optimizer.step()
        if self._lr_scheduler is not None and not skipped:
            if metric is not None:
                self._lr_scheduler.step(metric)
            elif (
                    self._schedule_steps is None
                    or self._lr_scheduler.last_epoch < self._schedule_steps
            ):
                # A schedule is defined over the steps it was built for.
                # Past them -- a run extended without replacing it -- the lr
                # holds at the schedule's final value: stepping on would
                # wrap a cosine back up to its base lr, and OneCycleLR
                # refuses outright.
                self._lr_scheduler.step()
        self._optimizer.zero_grad()

    def record_grad_norm(self, norm: float) -> None:
        self._grad_norm = float(norm)

    @property
    def grad_norm(self) -> float | None:
        """The last gradient norm measured, before clipping; None if none."""
        return self._grad_norm

    @property
    def current_lrs(self) -> list[float] | None:
        """Current learning rate of each param group; None when inactive."""
        if self._optimizer is None:
            return None
        return [group["lr"] for group in self._optimizer.param_groups]


@requires_resource("optimizer")
@hook("forward_context", session_type="training")
class ForwardContext(LifecycleHook):
    """Open each iteration on the optimizer before the forward pass runs."""

    call_every = 1

    @override
    def pre_session(self, session: Session) -> None:
        pass

    @override
    def pre_iteration_callback(self, session: Session) -> None:
        self.get_dependency("optimizer").begin_iteration(session.iteration)

    @override
    def post_iteration_callback(self, session: Session) -> None:
        # `backward` has closed them already; this covers an iteration whose
        # loss was never backpropagated.
        self.get_dependency("optimizer").close_contexts()

    @override
    def post_session(self, session: Session) -> None:
        self.get_dependency("optimizer").close_contexts()


@dataclass
class BackwardConfig:
    loss_key: str = "loss"

    def __post_init__(self):
        if not isinstance(self.loss_key, str) or not self.loss_key:
            raise ValueError(
                f"loss_key must be a non-empty string; got {self.loss_key!r}"
            )


@requires_hook("forward_context")
@requires_resource("optimizer")
@step("backward", session_type="training")
class Backward(Step):
    """Backpropagate the loss another step wrote under `loss_key`.

    It reads that key, so it runs after whichever step declares writing it.
    """

    config_schema = BackwardConfig

    @override
    def context_reads(self) -> dict[str, str]:
        return {"loss": self._cfg.loss_key}

    @override
    def run(self, session: Session, loss: Any) -> None:
        self.get_dependency("optimizer").backward(loss)


@requires_resource("optimizer")
class GradientProcessor(Step, ExtendableComponent):
    """A step that edits gradients between `backward` and `optimizer_step`.

    `process` runs only on iterations that step, after the gradients are
    complete and unscaled. A subclass requires the stage it follows and is
    put in the chain by binding `optimizer_step` to it; one that ends up
    after the step is refused before any step is taken.

    A subclass with a `config_schema` can be reconfigured by a session
    extension: it holds no state beyond its configuration.
    """

    @override
    def run(self, session: Session) -> None:
        optimizer = self.get_dependency("optimizer")
        if not optimizer.is_boundary:
            return
        self.process(session, optimizer.named_gradients())
        optimizer.mark_processed(self)

    @abstractmethod
    def process(
            self,
            session: Session,
            named_parameters: list[tuple[str, nn.Parameter]],
    ) -> None:
        """Edit `parameter.grad` in place for the given parameters."""
        raise NotImplementedError

    @override
    def apply_extension_config(
            self,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        if type(self).config_schema is None:
            raise ValueError(
                f"{self._component_name()} has no configuration to extend"
            )
        self._parse_config_schema(config)


@dataclass
class FreezeGradientsConfig:
    rules: tuple = ()

    def __post_init__(self):
        rules = []
        for index, rule in enumerate(self.rules):
            path = f"freeze_gradients.rules[{index}]"
            rule = _require_mapping(rule, path)
            unknown = _unknown_keys(rule, frozenset({"match", "until_iteration"}))
            if unknown:
                raise ValueError(f"Unknown {path} fields: {', '.join(unknown)}")
            until = rule.get("until_iteration")
            if isinstance(until, bool) or not isinstance(until, int) or until < 0:
                raise ValueError(
                    f"{path}.until_iteration must be a non-negative integer; "
                    f"got {until!r}"
                )
            rules.append({
                "match": _patterns(rule.get("match"), f"{path}.match"),
                "until_iteration": until,
            })
        self.rules = tuple(rules)


@requires_step("backward")
@step("freeze_gradients", session_type="training")
class FreezeGradients(GradientProcessor):
    """Drop the gradients of matching parameters until an iteration.

    A parameter whose gradient is None is skipped by the optimizer entirely,
    weight decay and moment updates included. With no rules it does nothing.
    """

    config_schema = FreezeGradientsConfig

    def __init__(self, config=None):
        super().__init__(config)
        self._patterns_checked = False

    @override
    def apply_extension_config(self, config, changed_paths) -> None:
        super().apply_extension_config(config, changed_paths)
        self._patterns_checked = False

    @override
    def run(self, session: Session) -> None:
        if not self._patterns_checked and self._cfg.rules:
            names = self.get_dependency("optimizer").parameter_names()
            for index, rule in enumerate(self._cfg.rules):
                _check_patterns_match(
                    rule["match"], names,
                    f"freeze_gradients.rules[{index}].match",
                )
            self._patterns_checked = True
        super().run(session)

    @override
    def process(self, session, named_parameters) -> None:
        active = [
            rule["match"] for rule in self._cfg.rules
            if session.iteration <= rule["until_iteration"]
        ]
        if not active:
            return
        for name, parameter in named_parameters:
            if any(_matches(name, patterns) for patterns in active):
                parameter.grad = None


@dataclass
class ClipGradientsConfig:
    max_norm: float | None = None
    norm_type: float = 2.0
    track_norm: bool = False

    def __post_init__(self):
        if self.max_norm is not None:
            if (
                    isinstance(self.max_norm, bool)
                    or not isinstance(self.max_norm, (int, float))
                    or not math.isfinite(self.max_norm)
                    or self.max_norm <= 0
            ):
                raise ValueError(
                    "max_norm must be a finite positive number or null; got "
                    f"{self.max_norm!r}"
                )
            self.max_norm = float(self.max_norm)
        if (
                isinstance(self.norm_type, bool)
                or not isinstance(self.norm_type, (int, float))
                or math.isnan(self.norm_type)
                or self.norm_type <= 0
        ):
            # inf is valid: the largest absolute gradient.
            raise ValueError(
                "norm_type must be a positive number (inf for the largest "
                f"absolute gradient); got {self.norm_type!r}"
            )
        self.norm_type = float(self.norm_type)
        if not isinstance(self.track_norm, bool):
            raise ValueError(
                f"track_norm must be a boolean; got {self.track_norm!r}"
            )


@requires_step("freeze_gradients")
@step("clip_gradients", session_type="training")
class ClipGradients(GradientProcessor):
    """Clip the total gradient norm, or only measure it.

    The norm before clipping is recorded on the optimizer as `grad_norm`.
    With neither `max_norm` nor `track_norm` it does nothing.
    """

    config_schema = ClipGradientsConfig

    @override
    def process(self, session, named_parameters) -> None:
        cfg = self._cfg
        if cfg.max_norm is None and not cfg.track_norm:
            return
        parameters = [parameter for _, parameter in named_parameters]
        if cfg.max_norm is not None:
            norm = nn.utils.clip_grad_norm_(
                parameters, cfg.max_norm, norm_type=cfg.norm_type,
            )
        else:
            norm = nn.utils.get_total_norm(
                [parameter.grad for parameter in parameters],
                norm_type=cfg.norm_type,
            )
        self.get_dependency("optimizer").record_grad_norm(norm)


@requires_step("clip_gradients")
@requires_resource("optimizer")
@step("optimizer_step", session_type="training")
class OptimizerStep(Step):
    """Step the optimizer and its schedule on iterations that end a group."""

    @override
    def context_reads(self) -> dict[str, str]:
        # A metric-driven schedule reads its metric here, so a session that
        # never writes it is rejected when it is built.
        if not self.has_dependency("optimizer"):
            return {}
        metric_key = self.get_dependency("optimizer").metric_key
        return {} if metric_key is None else {"metric": metric_key}

    @override
    def run(self, session: Session, metric: Any = None) -> None:
        optimizer = self.get_dependency("optimizer")
        if not optimizer.is_boundary:
            return
        optimizer.check_processed(
            processor for processor in session.get_all_steps()
            if isinstance(processor, GradientProcessor)
            and processor.has_dependency("optimizer")
            and processor.get_dependency("optimizer") is optimizer
        )
        optimizer.step(metric)


__all__ = [
    "Backward",
    "ClipGradients",
    "ForwardContext",
    "FreezeGradients",
    "GradientProcessor",
    "OptimizerResource",
    "OptimizerStep",
]
