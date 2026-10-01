"""Record one forward call of a module: which torch operation ran in which
submodule, and which tensor fed which.

Forward hooks on every submodule keep a stack of the module running; a
`TorchFunctionMode`, active only for that call, sees each torch operation,
runs it unchanged and notes the tensors it read and produced. Tensors are
linked by identity at the moment they are used, and only their identity and
shape are kept, never the tensors themselves.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten

from training_framework.components.builtin.diagram.graph import Graph, Node

SHAPE_OPS = frozenset({
    "view", "reshape", "flatten", "unflatten", "permute", "transpose", "t",
    "contiguous", "unsqueeze", "squeeze", "expand", "expand_as", "view_as",
    "reshape_as", "movedim", "swapaxes", "__getitem__", "chunk", "split",
    "unbind", "narrow", "to", "type_as", "float", "half", "bfloat16", "clone",
    "detach",
})
"""Operations that only rearrange or relabel a tensor; folded into the edge
by default."""

Source = tuple[str, Any]
"""Where a tensor came from: ("input", name) or ("op", index)."""


@dataclass
class Op:
    name: str
    path: str
    """Dotted module path the operation ran in; "" for the root itself."""
    inputs: list[tuple[Source, tuple[int, ...]]] = field(default_factory=list)


@dataclass
class Recording:
    """What one forward call did."""

    root_type: str
    ops: list[Op]
    inputs: list[tuple[str, tuple[int, ...]]]
    outputs: list[tuple[str, Source, tuple[int, ...]]]
    module_types: dict[str, str]
    """Module path -> class name, for every submodule."""


def _tensors(value) -> list[torch.Tensor]:
    leaves, _ = tree_flatten(value)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(size) for size in tensor.shape)


class _Mode(TorchFunctionMode):
    def __init__(self, recorder: "ForwardRecorder") -> None:
        super().__init__()
        self._recorder = recorder

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._recorder._record_op(func, args, kwargs, result)
        return result


class ForwardRecorder:
    """Record the next forward call of `root`, then remove itself.

    `on_done(recording, error)` is called once, right after that call (also
    when it raised); `error` is what went wrong while recording, if anything.
    The forward call itself is never changed: every hook returns None and
    every operation runs as called.
    """

    def __init__(
            self,
            root: nn.Module,
            on_done: Callable[[Recording | None, BaseException | None], None],
    ) -> None:
        self._root = root
        self._on_done = on_done
        self._paths = {module: name for name, module in root.named_modules()}
        self._stack: list[str] = []
        self._producers: dict[int, Source] = {}
        self._ops: list[Op] = []
        self._inputs: list[tuple[str, tuple[int, ...]]] = []
        # Every submodule's class, including containers that never run a
        # forward of their own (a ModuleDict holding the modules that do).
        self._module_types: dict[str, str] = {
            name: type(module).__name__
            for name, module in root.named_modules() if name
        }
        self._handles: list[Any] = []
        self._mode: _Mode | None = None
        self._error: BaseException | None = None
        self._finished = False
        self._root_handles = [
            root.register_forward_pre_hook(self._root_pre, with_kwargs=True),
            root.register_forward_hook(
                self._root_post, with_kwargs=True, always_call=True,
            ),
        ]

    @property
    def finished(self) -> bool:
        return self._finished

    def remove(self) -> None:
        """Stop waiting for a call; nothing is reported."""
        self._finished = True
        self._release()

    # -- hooks (each returns None: a value would replace the module's) ---------

    def _root_pre(self, module, args, kwargs) -> None:
        if self._finished or self._mode is not None:
            return None
        try:
            self._record_inputs(args, kwargs)
            for submodule, path in self._paths.items():
                if submodule is self._root:
                    continue
                self._handles.append(submodule.register_forward_pre_hook(
                    self._enter(path),
                ))
                self._handles.append(submodule.register_forward_hook(
                    self._leave, always_call=True,
                ))
            self._mode = _Mode(self)
            self._mode.__enter__()
        except Exception as error:  # never break the forward pass
            self._error = error
            self._finish(None)
        return None

    def _root_post(self, module, args, kwargs, output) -> None:
        if self._finished or self._mode is None:
            return None
        recording = None
        try:
            self._mode.__exit__(None, None, None)
            self._mode = None
            self._release()
            if self._error is None:
                outputs = []
                for index, tensor in enumerate(_tensors(output)):
                    source = self._producers.get(id(tensor))
                    if source is not None:
                        outputs.append((f"output[{index}]", source, _shape(tensor)))
                if len(outputs) == 1:
                    outputs[0] = ("output", *outputs[0][1:])
                recording = Recording(
                    root_type=type(self._root).__name__,
                    ops=self._ops,
                    inputs=self._inputs,
                    outputs=outputs,
                    module_types=self._module_types,
                )
        except Exception as error:
            self._error = error
            recording = None
        self._finish(recording)
        return None

    def _enter(self, path: str):
        def hook(module, args) -> None:
            self._stack.append(path)
            return None
        return hook

    def _leave(self, module, args, output) -> None:
        if self._stack:
            self._stack.pop()
        return None

    # -- recording -----------------------------------------------------------

    def _record_inputs(self, args, kwargs) -> None:
        """Name each input tensor after the argument it came in: keyword
        arguments collected by `**kwargs` by their own key, a mapping's
        entries as `name.key`, a sequence's as `name[i]`."""
        named: list[tuple[str, Any]] = []
        try:
            signature = inspect.signature(self._root.forward)
            bound = signature.bind_partial(*args, **kwargs).arguments
            for name, value in bound.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    named.extend(value.items())
                else:
                    named.append((name, value))
        except (TypeError, ValueError):
            named = [(f"arg{i}", value) for i, value in enumerate(args)]
            named.extend(kwargs.items())
        for name, value in named:
            entries = (
                [(f"{name}.{key}", item) for key, item in value.items()]
                if isinstance(value, Mapping) else [(name, value)]
            )
            for label, item in entries:
                tensors = _tensors(item)
                for index, tensor in enumerate(tensors):
                    tensor_label = (
                        label if len(tensors) == 1 else f"{label}[{index}]"
                    )
                    self._producers[id(tensor)] = ("input", tensor_label)
                    self._inputs.append((tensor_label, _shape(tensor)))

    def _record_op(self, func, args, kwargs, result) -> None:
        if self._error is not None:
            return
        try:
            outputs = _tensors(result)
            if not outputs:
                return  # attribute reads, sizes, comparisons to Python values
            name = getattr(func, "__name__", None) or str(func)
            op = Op(name=name, path=self._stack[-1] if self._stack else "")
            for tensor in _tensors((args, kwargs)):
                source = self._producers.get(id(tensor))
                if source is not None:
                    op.inputs.append((source, _shape(tensor)))
            index = len(self._ops)
            self._ops.append(op)
            for tensor in outputs:
                self._producers[id(tensor)] = ("op", index)
        except Exception as error:
            self._error = error

    def _release(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._finished:
            for handle in self._root_handles:
                handle.remove()
            self._root_handles.clear()

    def _finish(self, recording: Recording | None) -> None:
        if self._mode is not None:
            try:
                self._mode.__exit__(None, None, None)
            finally:
                self._mode = None
        self._finished = True
        self._release()
        self._producers.clear()
        self._on_done(recording, self._error)


# -- from a recording to a graph -----------------------------------------------


def _shape_label(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(size) for size in shape) + "]"


def build_model_graph(
        recording: Recording,
        *,
        title: str,
        depth: int,
        show_shape_ops: bool,
        collapse_repeats: bool,
        component_names: dict[str, str] | None = None,
) -> Graph:
    """The data flow of `recording` at `depth` module levels below the root.

    An operation that ran inside a module `depth` levels down or deeper
    belongs to that module's node; one that ran in a shallower module whose
    deeper modules are shown is an operation node in that module's cluster.
    `component_names` (module path -> component instance) labels the modules
    that are framework components.
    """
    component_names = component_names or {}
    graph = Graph(title=title)
    shown_parents: set[str] = set()
    for op in recording.ops:
        parts = op.path.split(".") if op.path else []
        for i in range(1, min(len(parts), depth)):
            shown_parents.add(".".join(parts[:i]))

    def module_label(path: str) -> str:
        type_name = recording.module_types.get(path, "")
        label = f"{path.rsplit('.', 1)[-1]}\n{type_name}"
        if path in component_names:
            label += f"\n= {component_names[path]}"
        return label

    def cluster_of(parts: list[str]) -> tuple[str, ...]:
        return tuple(".".join(parts[:i]) for i in range(1, len(parts) + 1))

    for path in shown_parents:
        graph.cluster_labels[path] = module_label(path).replace("\n", " ")

    for name, shape in recording.inputs:
        graph.add_node(Node(f"in:{name}", f"{name}\n{_shape_label(shape)}", "input"))

    keys: list[str] = []
    for index, op in enumerate(recording.ops):
        parts = op.path.split(".") if op.path else []
        if parts and not (len(parts) < depth and op.path in shown_parents):
            module_path = ".".join(parts[:depth])
            key = f"mod:{module_path}"
            graph.add_node(Node(
                key, module_label(module_path), "module",
                cluster_of(parts[:depth][:-1]),
            ))
        else:
            key = f"op:{index}"
            graph.add_node(Node(key, op.name, "op", cluster_of(parts)))
        keys.append(key)

    def key_of(source: Source) -> str:
        kind, value = source
        return f"in:{value}" if kind == "input" else keys[value]

    for index, op in enumerate(recording.ops):
        for source, shape in op.inputs:
            graph.add_edge(key_of(source), keys[index], _shape_label(shape))
    for name, source, shape in recording.outputs:
        graph.add_node(Node(f"out:{name}", f"{name}\n{_shape_label(shape)}", "output"))
        graph.add_edge(key_of(source), f"out:{name}", _shape_label(shape))

    if not show_shape_ops:
        _fold_shape_ops(graph)
    if collapse_repeats:
        _collapse_repeats(graph, recording.module_types)
    return graph


def _fold_shape_ops(graph: Graph) -> None:
    for key in [k for k, n in graph.nodes.items() if n.kind == "op" and n.label in SHAPE_OPS]:
        predecessors = graph.predecessors(key)
        successors = [(t, graph.edges[(key, t)]) for t in graph.successors(key)]
        graph.remove_node(key)
        for source in predecessors:
            for target, label in successors:
                graph.add_edge(source, target, label)


_REPEAT = re.compile(r"\A(?P<prefix>.*?)(?P<index>\d+)\Z")


def _collapse_repeats(graph: Graph, module_types: dict[str, str]) -> None:
    """Draw a chain of numbered sibling modules of one type once:
    `layers.0` -> `layers.1` -> ... becomes `layers.0-5 ... x6`."""
    groups: dict[tuple, list[tuple[int, str]]] = {}
    for key, node in graph.nodes.items():
        if node.kind != "module":
            continue
        path = key[len("mod:"):]
        match = _REPEAT.match(path)
        if match is None or not match["prefix"].endswith("."):
            continue
        signature = (match["prefix"], module_types.get(path), node.cluster)
        groups.setdefault(signature, []).append((int(match["index"]), key))
    for (prefix, type_name, cluster), members in groups.items():
        members.sort()
        indices = [index for index, _ in members]
        keys = [key for _, key in members]
        if len(keys) < 2 or indices != list(range(indices[0], indices[0] + len(keys))):
            continue
        chained = all(
            graph.successors(keys[i]) == [keys[i + 1]]
            and graph.predecessors(keys[i + 1]) == [keys[i]]
            for i in range(len(keys) - 1)
        )
        if not chained:
            continue
        first, last = keys[0], keys[-1]
        inbound = [(s, graph.edges[(s, first)]) for s in graph.predecessors(first)]
        outbound = [(t, graph.edges[(last, t)]) for t in graph.successors(last)]
        order = graph.nodes[first].order
        for key in keys:
            graph.remove_node(key)
        name = f"{prefix.rstrip('.').rsplit('.', 1)[-1]}.{indices[0]}-{indices[-1]}"
        merged = f"mod:{prefix}{indices[0]}-{indices[-1]}"
        graph.add_node(Node(
            merged, f"{name}\n{type_name} x{len(keys)}", "module", cluster,
        ))
        graph.nodes[merged].order = order
        for source, label in inbound:
            graph.add_edge(source, merged, label)
        for target, label in outbound:
            graph.add_edge(merged, target, label)
