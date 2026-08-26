import os
from collections.abc import Mapping
from typing import Any, cast

from training_framework.session.base import Session
from training_framework.session.config import SessionMode


class AnalysisSession(Session):
    """Analysis workflow driven by a trained-model session checkpoint."""

    @classmethod
    def _session_mode(cls) -> SessionMode:
        return SessionMode.ANALYSIS

    @classmethod
    def _default_component_configs(cls) -> Mapping[str, Mapping]:
        return {
            "trained_model": {},
            "logger": {"log_every": 10},
        }

    def __init__(
            self,
            config: dict,
            *,
            model_checkpoint_path: str | os.PathLike[str],
    ):
        self._model_checkpoint_path = self._validate_model_checkpoint_path(
            model_checkpoint_path,
        )
        super().__init__(config)
        self._init_args = {
            "args": (config,),
            "kwargs": {
                "model_checkpoint_path": self._model_checkpoint_path,
            },
        }

    @staticmethod
    def _validate_model_checkpoint_path(
            checkpoint_path: str | os.PathLike[str] | None,
    ) -> str:
        if checkpoint_path is None:
            raise ValueError("model_checkpoint_path is required")
        try:
            normalized_path = os.fspath(cast(Any, checkpoint_path))
        except TypeError as error:
            raise TypeError(
                "model_checkpoint_path must be a string or path-like object"
            ) from error
        if not isinstance(normalized_path, str):
            raise TypeError("model_checkpoint_path must resolve to a string path")
        if not os.path.isfile(normalized_path):
            raise FileNotFoundError(
                f"Model checkpoint does not exist: {normalized_path}"
            )
        return normalized_path

    @property
    def model_checkpoint_path(self) -> str:
        return self._model_checkpoint_path

    def _get_mode_state(self) -> dict[str, Any]:
        return {"model_checkpoint_path": self._model_checkpoint_path}

    def _restore_mode_state(self, state: Mapping[str, Any]) -> None:
        self._model_checkpoint_path = self._validate_model_checkpoint_path(
            state.get("model_checkpoint_path"),
        )
