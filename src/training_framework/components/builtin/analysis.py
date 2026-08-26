from __future__ import annotations

from typing import TYPE_CHECKING, Any

from training_framework.components import Resource, hook, resource
from training_framework.components.builtin.checkpointing import Checkpointer
from training_framework.components.builtin.observability import Logger
from training_framework.util import context_entry, context_exit, requires_context

if TYPE_CHECKING:
    from training_framework.session import AnalysisSession, Session


@resource("trained_model", mode="analysis")
class TrainedModel(Resource):
    """Expose the model resource restored from a training-session checkpoint."""

    def __init__(self, config: dict):
        self._config = config
        self._source_session = None
        self._model: Any = None

    @property
    @requires_context
    def model(self):
        return self._model

    @context_entry
    def setup(self, session: AnalysisSession) -> Any:
        from training_framework.session import Session, SessionMode

        source_session = Checkpointer.load_checkpoint(
            session.model_checkpoint_path,
            map_location="cpu",
        )
        if not isinstance(source_session, Session):
            raise TypeError(
                "Analysis model checkpoint must contain a framework Session"
            )
        if source_session.mode is not SessionMode.TRAINING:
            raise ValueError(
                "Analysis model checkpoint must contain a training-mode session"
            )

        try:
            model = source_session.get_resource("model")
        except KeyError as error:
            raise ValueError(
                "Training checkpoint does not contain a resolvable 'model' resource"
            ) from error

        if not callable(getattr(model, "to", None)):
            raise TypeError("The checkpoint 'model' resource must provide to(device)")
        if not callable(getattr(model, "eval", None)):
            raise TypeError("The checkpoint 'model' resource must provide eval()")

        moved_model = model.to(session.device)
        self._model = model if moved_model is None else moved_model
        self._model.eval()
        self._source_session = source_session

    @context_exit
    def teardown(self, session: AnalysisSession) -> None:
        self._model = None
        self._source_session = None


@hook("logger", mode="analysis")
class AnalysisLogger(Logger):
    """Default progress logger for analysis sessions."""

    def pre_iteration_callback(self, session: Session) -> None:
        self.print(
            f"Analysis iteration {session.iteration}/"
            f"{session.session_config.max_iterations}"
        )
