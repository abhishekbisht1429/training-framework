"""`model_diagram`: pictures of the model being run, in the session dir."""

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    SessionHook,
    hook,
    rank_zero_only,
    requires_resource,
)
from training_framework.components.builtin.diagram.capture import (
    ForwardRecorder,
    Recording,
    build_model_graph,
)
from training_framework.components.builtin.diagram.graph import (
    GRAPHVIZ_HINT,
    Graph,
    valid_stem,
    write_diagram,
)
from training_framework.components.builtin.diagram.wiring import (
    build_session_graph,
    build_wiring_graph,
    component_module_paths,
)

if TYPE_CHECKING:
    from training_framework.session import Session

FORMATS = ("png", "svg", "pdf")
SCOPES = ("model", "session", "both")


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false; got {value!r}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


@dataclass
class ModelDiagramConfig:
    depth: Any = 2
    show_shape_ops: Any = False
    collapse_repeats: Any = True
    component_wiring: Any = True
    scope: Any = "both"
    formats: Any = ("png", "svg")
    dpi: Any = 150
    file: Any = "model_diagram"
    graphviz_dot: Any = None

    def __post_init__(self):
        self.depth = _positive_int(self.depth, "depth")
        self.dpi = _positive_int(self.dpi, "dpi")
        self.show_shape_ops = _flag(self.show_shape_ops, "show_shape_ops")
        self.collapse_repeats = _flag(self.collapse_repeats, "collapse_repeats")
        self.component_wiring = _flag(self.component_wiring, "component_wiring")
        if self.scope not in SCOPES:
            raise ValueError(
                f"scope must be one of {list(SCOPES)}; got {self.scope!r}"
            )
        formats = self.formats
        if (
                not isinstance(formats, (list, tuple)) or not formats
                or any(fmt not in FORMATS for fmt in formats)
        ):
            raise ValueError(
                f"formats must be a non-empty list of {list(FORMATS)}; "
                f"got {formats!r}"
            )
        self.formats = tuple(dict.fromkeys(formats))
        if self.graphviz_dot is not None and (
                not isinstance(self.graphviz_dot, str) or not self.graphviz_dot
        ):
            raise ValueError(
                "graphviz_dot must be the path of Graphviz's `dot`; got "
                f"{self.graphviz_dot!r}"
            )
        if not isinstance(self.file, str) or not valid_stem(self.file):
            raise ValueError(
                "file must be a file name without a directory (letters, "
                f"digits, '_', '-', '.'); got {self.file!r}"
            )


class _RunState:
    """What one run of a `ModelDiagram` holds: the recorder waiting for the
    model's first forward call, where it writes, what it wrote.

    It belongs to one hook and never travels: a pickle or deep copy of the
    hook gets an empty one (`__reduce__`), and a shallow copy, which shares
    the hook's `__dict__`, is given its own on first use (`owner`). A copy
    therefore starts not recording, and the original keeps its recorder.
    """

    def __init__(self, owner: "ModelDiagram | None" = None) -> None:
        self.owner = weakref.ref(owner) if owner is not None else None
        self.recorder: ForwardRecorder | None = None
        self.directory: str | None = None
        self.written: list[str] = []

    def __reduce__(self):
        return (_RunState, ())


_RUN_STATE_ATTR = "_model_diagram_run"


@rank_zero_only
@requires_resource("model")
@hook("model_diagram", session_type=TRAINING_SESSION_TYPE)
@hook("model_diagram", session_type=ANALYSIS_SESSION_TYPE)
class ModelDiagram(SessionHook):
    """Draw the model being run, and how its components are wired.

    The model diagram is taken from the model's first forward call in the
    run: its inputs, the modules down to `depth` levels and the operations
    between them, linked by the tensors that flow, with their shapes. The
    component wiring is drawn when the session starts, for the model and
    what it is wired to (`scope: model`, `<file>_components.*`), for every
    component of the session (`scope: session`, `<file>_session.*`), or both
    (the default). Everything goes to the session dir as
    `.{png,svg,...,dot,mmd}`, on rank 0 only.

    Another model than the one bound to `model` -- a teacher, say -- is drawn
    by a second instance wired to it:
    `model_diagram#teacher: {dependencies_role_bindings: {model: teacher}}`.

    Nothing that goes wrong while recording or drawing stops the run: it is
    reported as a warning. The recording adds a little Python work to that
    one forward call and is removed right after it.
    """

    config_schema = ModelDiagramConfig

    @property
    def _run(self) -> _RunState:
        run = self.__dict__.get(_RUN_STATE_ATTR)
        if run is None or run.owner is None or run.owner() is not self:
            run = self.__dict__[_RUN_STATE_ATTR] = _RunState(self)
        return run

    @property
    def written(self) -> list[str]:
        """Files written so far in this run."""
        return list(self._run.written)

    def _stem(self, extra: str = "") -> str:
        suffix = self.instance_suffix
        return f"{self._cfg.file}{extra}" + (f"_{suffix}" if suffix else "")

    def pre_session(self, session: "Session") -> None:
        model = self.get_dependency("model")
        self._run.directory = session.session_config.session_dir
        scope = self._cfg.scope if self._cfg.component_wiring else None
        if scope in ("model", "both"):
            self._draw(
                lambda: build_wiring_graph(
                    model,
                    title=f"Component wiring of {model.name}",
                    imported=session.imported_components,
                ),
                self._stem("_components"),
            )
        if scope in ("session", "both"):
            self._draw(
                lambda: build_session_graph(
                    [
                        *session.get_all_resources(),
                        *session.get_all_hooks(),
                        *session.get_all_steps(),
                    ],
                    session.resolved_component_edges(),
                    title=f"Component wiring of the {session.session_type} session",
                    imported=session.imported_components,
                ),
                self._stem("_session"),
            )
        if not isinstance(model, nn.Module):
            self._warn(f"'{model.name}' is not an nn.Module; no model diagram")
            return
        if hasattr(model, "_orig_mod"):
            self._warn(
                f"'{model.name}' is compiled (torch.compile): the operations "
                "inside it cannot be seen, so no model diagram is drawn"
            )
            return
        self._run.recorder = ForwardRecorder(
            model,
            lambda recording, error: self._recorded(model, recording, error),
        )

    def post_session(self, session: "Session") -> None:
        recorder, self._run.recorder = self._run.recorder, None
        if recorder is not None and not recorder.finished:
            recorder.remove()
            self._warn(
                "the model never ran a forward call in this session; no "
                "model diagram"
            )

    def rollback_pre_session(self, session: "Session") -> None:
        recorder, self._run.recorder = self._run.recorder, None
        if recorder is not None:
            recorder.remove()

    def _recorded(
            self,
            model: nn.Module,
            recording: Recording | None,
            error: BaseException | None,
    ) -> None:
        if recording is None:
            self._warn(f"recording the forward call failed: {error!r}")
            return
        self._draw(
            lambda: build_model_graph(
                recording,
                title=f"{model.name} ({recording.root_type})",
                depth=self._cfg.depth,
                show_shape_ops=self._cfg.show_shape_ops,
                collapse_repeats=self._cfg.collapse_repeats,
                component_names=component_module_paths(model),
            ),
            self._stem(),
        )

    def _draw(self, build, stem: str) -> None:
        try:
            graph: Graph = build()
            written, fallback = write_diagram(
                graph, self._run.directory, stem, self._cfg.formats,
                self._cfg.dpi, self._cfg.graphviz_dot,
            )
        except Exception as error:  # a picture must not stop the run
            self._warn(f"drawing {stem} failed: {error!r}")
            return
        self._run.written.extend(written)
        print(
            f"{self.name}: wrote "
            + ", ".join(os.path.basename(path) for path in written)
            + f" to {self._run.directory}",
            flush=True,
        )
        if fallback is not None:
            self._warn(
                f"{fallback}; {stem} was drawn with matplotlib instead. "
                + GRAPHVIZ_HINT
            )

    def _warn(self, message: str) -> None:
        warnings.warn(f"{self.name}: {message}", RuntimeWarning, stacklevel=2)
