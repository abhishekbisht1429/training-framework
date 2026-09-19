from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from training_framework.components import Resource, resource
from training_framework.components.builtin.checkpointing import Checkpointer
from training_framework.util import requires_context

if TYPE_CHECKING:
    from training_framework.session import Session


@resource("trained_model")
class TrainedModel(Resource):
    """Expose the model resource restored from a training-session checkpoint."""

    def __init__(self, config: Mapping):
        super().__init__(config)
        if not isinstance(config, Mapping):
            raise TypeError("trained_model config must be a mapping")
        checkpoint_path = config.get("model_checkpoint_path")
        if checkpoint_path is None:
            raise ValueError(
                "trained_model.model_checkpoint_path is required"
            )
        try:
            normalized_path = os.fspath(cast(Any, checkpoint_path))
        except TypeError as error:
            raise TypeError(
                "trained_model.model_checkpoint_path must be a string or "
                "path-like object"
            ) from error
        if not isinstance(normalized_path, str):
            raise TypeError(
                "trained_model.model_checkpoint_path must resolve to a "
                "string path"
            )
        if not os.path.isfile(normalized_path):
            raise FileNotFoundError(
                f"Model checkpoint does not exist: {normalized_path}"
            )

        self._source_session = None
        self._model: Any = None
        self._model_checkpoint_path = normalized_path

    @property
    @requires_context
    def model(self):
        return self._model

    def setup(self, session: Session) -> Any:
        from training_framework.session import (
            Session as FrameworkSession,
            TRAINING_SESSION_TYPE,
        )

        source_session = Checkpointer.load_checkpoint(
            self._model_checkpoint_path,
            map_location="cpu",
            # The weights are wanted, not the run they came from: restoring
            # the source session's RNG would reseed this one mid-setup.
            restore_rng=False,
        )
        if not isinstance(source_session, FrameworkSession):
            raise TypeError(
                "Analysis model checkpoint must contain a framework Session"
            )
        if source_session.session_type != TRAINING_SESSION_TYPE:
            raise ValueError(
                "Analysis model checkpoint must contain a training session"
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

    def teardown(self, session: Session) -> None:
        self._model = None
        self._source_session = None
