from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

from torch import empty, nn, optim

from training_framework.components import (
    ExtendableComponent,
    StatefulLifeCycleHook,
)
from training_framework.components import hook, requires_resource

if TYPE_CHECKING:
    from training_framework.session import Session


_LEGACY_CONFIG_KEYS = frozenset({
    "learning_rate",
    "weight_decay",
    "warmup_iters",
})
_SPEC_KEYS = frozenset({"name", "kwargs"})
_SCHEDULER_KEYS = frozenset({"stages", "milestones", "metric_key"})
_RUNTIME_PLACEHOLDERS = frozenset({
    "$max_iterations",
    "$stage_iterations",
})


def _unknown_keys(config: Mapping, allowed: frozenset[str]) -> list[str]:
    return sorted(str(key) for key in set(config) - allowed)


def _supports_extension_path(path: tuple[str, ...]) -> bool:
    if path[:2] == ("optimizer", "kwargs") and len(path) >= 3:
        return True
    return bool(path) and path[0] == "lr_scheduler"


def _rebase_scheduler_lrs(state: Mapping, new_lr: float) -> None:
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

    A no-op for scheduler kinds with no ``base_lrs`` (e.g.
    ReduceLROnPlateau, which reads the optimizer's current lr directly).
    """
    schedulers = state.get("_schedulers")
    if schedulers:
        milestones = state.get("_milestones", [])
        last_epoch = state.get("last_epoch", 0)
        active_index = bisect_right(milestones, last_epoch)
        _rebase_scheduler_lrs(schedulers[active_index], new_lr)
        return
    if "base_lrs" not in state:
        return
    base_lrs = state["base_lrs"]
    last_lr = state.get("_last_lr", base_lrs)
    state["base_lrs"] = [
        base * (new_lr / current) if current else new_lr
        for base, current in zip(base_lrs, last_lr)
    ]
    state["_last_lr"] = [new_lr] * len(last_lr)


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


@hook("optimizer", session_type="training")
@requires_resource("ddp")
class OptimizerHook(StatefulLifeCycleHook, ExtendableComponent):

    def __init__(self, config):
        self.call_every = 1
        self._optimizer_spec, self._scheduler_config = self._normalize_config(
            config
        )
        self._optimizer = None
        self._lr_scheduler = None
        self._restored_state = None

    @staticmethod
    def _normalize_config(config):
        config = _require_mapping(config, "optimizer config")
        legacy = sorted(_LEGACY_CONFIG_KEYS & set(config))
        if legacy:
            raise ValueError(
                "Legacy optimizer fields are no longer supported: "
                f"{', '.join(legacy)}. Use optimizer.optimizer.name and "
                "optimizer.optimizer.kwargs; configure scheduling under "
                "optimizer.lr_scheduler."
            )
        unknown = _unknown_keys(
            config, frozenset({"optimizer", "lr_scheduler"})
        )
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
                "parameters are supplied by OptimizerHook"
            )

        scheduler_value = config.get("lr_scheduler")
        if scheduler_value is None:
            return optimizer_spec, None
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
                    "supplied by OptimizerHook"
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

        return optimizer_spec, {
            "stages": stages,
            "milestones": milestones,
            "metric_key": metric_key,
        }

    def _prepare_scheduler(self, optimizer, max_iterations):
        if self._scheduler_config is None:
            return None
        milestones = self._scheduler_config["milestones"]
        if max_iterations <= 0:
            raise ValueError(
                "A configured lr_scheduler requires max_iterations to be "
                "positive"
            )
        if milestones and milestones[-1] >= max_iterations:
            raise ValueError(
                "optimizer.lr_scheduler milestones must be less than "
                "session_config.max_iterations"
            )

        boundaries = [0, *milestones, max_iterations]
        schedulers = []
        for index, spec in enumerate(self._scheduler_config["stages"]):
            scheduler_class = _resolve_class(
                optim.lr_scheduler,
                spec["name"],
                optim.lr_scheduler.LRScheduler,
                "scheduler",
            )
            kwargs = _resolve_placeholders(
                spec["kwargs"],
                max_iterations=max_iterations,
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

    @override
    def pre_session(self, session: Session):
        ddp_model: nn.Module = session.get_resource("ddp")
        optimizer_class = _resolve_class(
            optim,
            self._optimizer_spec["name"],
            optim.Optimizer,
            "optimizer",
        )
        try:
            self._optimizer = optimizer_class(
                ddp_model.wrapped_model.parameters(),
                **deepcopy(self._optimizer_spec["kwargs"]),
            )
        except TypeError as error:
            raise ValueError(
                f"Invalid kwargs for optimizer "
                f"{self._optimizer_spec['name']!r}: {error}"
            ) from error
        scheduler_state = None
        if self._restored_state is not None:
            scheduler_state = self._restored_state.get("lr_scheduler_state")

        if scheduler_state is None:
            # No prior scheduler state will be layered on afterward (first
            # run, or an extension replaced the scheduler): restore the
            # optimizer's lr first, so a freshly constructed scheduler's
            # own initial-step application (e.g. a LinearLR/ConstantLR
            # warmup factor) uses the correct base — nothing will
            # overwrite it afterward.
            self._restore_optimizer_state()
            self._lr_scheduler = self._prepare_scheduler(
                self._optimizer, session.session_config.max_iterations
            )
            self._restore_scheduler_state()
        else:
            # An existing scheduler state is restored afterward via
            # load_state_dict, which only syncs the scheduler's own
            # counters (base_lrs, last_epoch, ...) and never touches
            # optimizer.param_groups. So the optimizer_state restore must
            # run last: it's what puts the correct current lr back after
            # the fresh scheduler's construction-time initial step
            # overwrote it with its own (stale, epoch-0) value.
            self._lr_scheduler = self._prepare_scheduler(
                self._optimizer, session.session_config.max_iterations
            )
            self._restore_optimizer_state()
            self._restore_scheduler_state()

    def _restore_optimizer_state(self):
        if self._restored_state is None:
            return
        optimizer_state = self._restored_state.get("optimizer_state")
        if optimizer_state is not None:
            self._optimizer.load_state_dict(optimizer_state)

    def _restore_scheduler_state(self):
        if self._restored_state is None:
            return
        scheduler_state = self._restored_state.get("lr_scheduler_state")
        if scheduler_state is not None:
            if self._lr_scheduler is None:
                raise ValueError(
                    "Cannot restore lr_scheduler_state without a configured "
                    "lr_scheduler"
                )
            self._lr_scheduler.load_state_dict(scheduler_state)
        self._restored_state = None

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

        optimizer_spec, scheduler_config = self._normalize_config(config)
        if optimizer_spec["name"] != self._optimizer_spec["name"]:
            raise ValueError(
                "Optimizer class cannot change during session extension"
            )
        if self._optimizer is not None:
            raise RuntimeError(
                "Optimizer configuration cannot change while the session "
                "is active"
            )

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

        scheduler_changed = scheduler_config != self._scheduler_config

        if self._restored_state is not None:
            optimizer_state = self._restored_state.get("optimizer_state")
            if optimizer_state is not None:
                for group in optimizer_state.get("param_groups", []):
                    for key in changed_kwarg_keys:
                        group[key] = deepcopy(optimizer_kwargs[key])

            if scheduler_changed:
                # New schedule shape/class: old scheduler state is not
                # guaranteed compatible, so restart schedule progress from
                # the extension point.
                self._restored_state["lr_scheduler_state"] = None
                if (
                        scheduler_config is not None
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
                    for group in optimizer_state.get("param_groups", []):
                        group["lr"] = deepcopy(effective_lr)
                        if "initial_lr" in group:
                            group["initial_lr"] = deepcopy(effective_lr)
            elif "lr" in changed_kwarg_keys:
                scheduler_state = self._restored_state.get("lr_scheduler_state")
                if scheduler_state is not None:
                    _rebase_scheduler_lrs(scheduler_state, optimizer_kwargs["lr"])

        self._optimizer_spec = optimizer_spec
        self._scheduler_config = scheduler_config

    @override
    def pre_iteration_callback(self, session: Session) -> None:
        self._optimizer.zero_grad()

    @override
    def post_iteration_callback(self, session: Session) -> None:
        loss = session.iteration_context["loss"]
        metric = None
        metric_key = (
            self._scheduler_config["metric_key"]
            if self._scheduler_config is not None
            else None
        )
        if metric_key is not None:
            try:
                metric = session.iteration_context[metric_key]
            except KeyError as error:
                raise KeyError(
                    f"Configured lr_scheduler metric {metric_key!r} is "
                    "missing from session.iteration_context"
                ) from error
        loss.backward()
        self._optimizer.step()
        if self._lr_scheduler is not None:
            if metric_key is None:
                self._lr_scheduler.step()
            else:
                self._lr_scheduler.step(metric)

    @override
    def post_session(self, session: Session):
        self._restored_state = self.get_state()
        self._optimizer = None
        self._lr_scheduler = None

    @override
    def rollback_pre_session(self, session: Session) -> None:
        self._optimizer = None
        self._lr_scheduler = None

    @override
    def set_state(self, state: Any) -> None:
        self._restored_state = deepcopy(state)
        if self._optimizer is not None:
            # The optimizer and scheduler already exist here (no fresh
            # scheduler construction involved), so restore order doesn't
            # matter the way it does in pre_session.
            self._restore_optimizer_state()
            self._restore_scheduler_state()

    @override
    def get_state(self) -> Any:
        if self._optimizer is None and self._lr_scheduler is None:
            if self._restored_state is not None:
                return deepcopy(self._restored_state)
            return {"optimizer_state": None, "lr_scheduler_state": None}
        return {
            "optimizer_state": (
                self._optimizer.state_dict() if self._optimizer else None
            ),
            "lr_scheduler_state": (
                self._lr_scheduler.state_dict()
                if self._lr_scheduler
                else None
            ),
        }
