from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from torch import nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    Resource,
    requires_resource,
    resource,
)
from training_framework.components.importing import import_module_class
from training_framework.util import requires_context

if TYPE_CHECKING:
    from training_framework.session import Session


@dataclass(frozen=True)
class LayerCapture:
    """One forward-hook observation of a single tracked layer.

    ``module`` is a live reference to the layer that ran (not a copy), so its
    parameters, e.g. ``capture.module.weight``, can be read alongside the
    captured tensors.
    """

    layer_name: str
    module_type: str
    input_args: tuple[Any, ...]
    input_kwargs: dict[str, Any]
    output: Any
    module: nn.Module


def _resolve_module_type(dotted_path: str) -> type[nn.Module]:
    if not isinstance(dotted_path, str) or "." not in dotted_path:
        raise ValueError(
            "layer_inspector.module_types entries must be fully-qualified "
            f"dotted paths (e.g. 'torch.nn.MultiheadAttention'); got "
            f"{dotted_path!r}"
        )
    return import_module_class(
        dotted_path,
        "layer_inspector.module_types entry",
        example="torch.nn.MultiheadAttention",
    )


@requires_resource("trained_model")
@resource("layer_inspector", session_type=ANALYSIS_SESSION_TYPE)
class LayerInspector(Resource):
    """Automate discovery and forward-pass capture of selected model layers.

    Matches layers of `trained_model.model` by name pattern and/or module
    type, registers a forward hook on each match, and exposes the current
    iteration's captured input/output as `captures`. Interpreting those
    captures (heatmaps, statistics, or anything else) is left entirely to
    the user's own analysis `Step`.
    """

    _CAPTURE_CONTEXT_KEY = "layer_inspector_captures"

    def __init__(self, config: Mapping) -> None:
        super().__init__(config)
        if not isinstance(config, Mapping):
            raise TypeError("layer_inspector config must be a mapping")

        name_patterns = config.get("name_patterns") or []
        module_type_paths = config.get("module_types") or []
        if not isinstance(name_patterns, list) or not all(
                isinstance(pattern, str) for pattern in name_patterns
        ):
            raise TypeError(
                "layer_inspector.name_patterns must be a list of strings"
            )
        if not isinstance(module_type_paths, list) or not all(
                isinstance(path, str) for path in module_type_paths
        ):
            raise TypeError(
                "layer_inspector.module_types must be a list of strings"
            )
        if not name_patterns and not module_type_paths:
            raise ValueError(
                "layer_inspector requires at least one of 'name_patterns' "
                "or 'module_types' to select layers"
            )

        try:
            self._name_patterns = [
                re.compile(pattern) for pattern in name_patterns
            ]
        except re.error as error:
            raise ValueError(
                "layer_inspector.name_patterns contains an invalid regex: "
                f"{error}"
            ) from error
        self._module_types = tuple(
            _resolve_module_type(path) for path in module_type_paths
        )
        self._always_call = bool(config.get("always_call", False))

        self._matched_layer_names: tuple[str, ...] = ()
        self._layers: dict[str, nn.Module] = {}
        self._handles: dict[str, Any] = {}
        self._session: Any = None

    @property
    def matched_layer_names(self) -> tuple[str, ...]:
        """The full set of layer names selected during `setup`."""
        return self._matched_layer_names

    @property
    def layers(self) -> Mapping[str, nn.Module]:
        """Read-only mapping of each selected layer name to its live module.

        Use it to inspect a layer's parameters directly, e.g.
        ``inspector.layers[name].weight`` or ``named_parameters()``; no
        forward pass is needed. Populated by `setup` in `matched_layer_names`
        order and emptied by `teardown`. The modules are the trained model's
        own, so avoid modifying them in place.
        """
        return MappingProxyType(self._layers)

    @property
    @requires_context
    def captures(self) -> dict[str, list[LayerCapture]]:
        """Captures observed so far this iteration, keyed by layer name.

        Sparse: only layers that actually ran a forward pass this iteration
        have an entry. Cleared automatically at every iteration boundary by
        the session (backed by `session.iteration_context`).
        """
        return self._session.iteration_context.setdefault(
            self._CAPTURE_CONTEXT_KEY, {}
        )

    def last_capture(self, layer_name: str) -> LayerCapture | None:
        """The most recent capture of `layer_name` this iteration, if any.

        Convenience for the common one-forward-pass-per-iteration case; use
        `captures[layer_name]` when a layer may run several times. Returns
        None if the layer did not run this iteration, and raises KeyError if
        `layer_name` is not one of `matched_layer_names`.
        """
        captures = self.captures
        if layer_name not in self._layers:
            raise KeyError(
                f"'{layer_name}' is not a layer selected by layer_inspector; "
                f"selected layers: {list(self._matched_layer_names)}"
            )
        layer_captures = captures.get(layer_name)
        return layer_captures[-1] if layer_captures else None

    def _selects(self, name: str, module: nn.Module) -> bool:
        if self._module_types and isinstance(module, self._module_types):
            return True
        return any(pattern.search(name) for pattern in self._name_patterns)

    def setup(self, session: "Session") -> None:
        model = self.get_dependency("trained_model").model
        matched = {
            name: module
            for name, module in model.named_modules()
            if self._selects(name, module)
        }
        if not matched:
            raise ValueError(
                "layer_inspector matched no layers in the trained model; "
                f"name_patterns={[p.pattern for p in self._name_patterns]}, "
                "module_types="
                f"{[t.__qualname__ for t in self._module_types]}"
            )

        self._session = session
        try:
            for name, module in matched.items():
                self._handles[name] = module.register_forward_hook(
                    self._make_hook(name),
                    with_kwargs=True,
                    always_call=self._always_call,
                )
        except Exception:
            self._release_partial_setup()
            raise

        self._matched_layer_names = tuple(matched.keys())
        self._layers = matched

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, args, kwargs, output):
            capture = LayerCapture(
                layer_name=layer_name,
                module_type=type(module).__qualname__,
                input_args=args,
                input_kwargs=dict(kwargs),
                output=output,
                module=module,
            )
            self.captures.setdefault(layer_name, []).append(capture)

        return hook

    def rollback_setup(self, session: "Session") -> None:
        self._release_partial_setup()

    def teardown(self, session: "Session") -> None:
        self._release_partial_setup()

    def _release_partial_setup(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles = {}
        self._matched_layer_names = ()
        self._layers = {}
        self._session = None
