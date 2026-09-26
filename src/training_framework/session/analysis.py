from collections.abc import Mapping

from training_framework.session.base import Session
from training_framework.session.config import ANALYSIS_SESSION_TYPE
from training_framework.session.registry import register_session_type


@register_session_type(ANALYSIS_SESSION_TYPE)
class AnalysisSession(Session):
    """Analysis workflow driven by a trained-model session checkpoint."""

    @classmethod
    def _default_component_configs(cls) -> Mapping[str, Mapping]:
        return {
            "trained_model": {},
            "logger": {"log_every": 10},
        }

    def __init__(self, config: dict):
        if isinstance(config, Mapping) and "model_checkpoint_path" in config:
            raise ValueError(
                "The top-level 'model_checkpoint_path' entry is no longer "
                "supported. Configure "
                "'trained_model.model_checkpoint_path' instead."
            )
        super().__init__(config)
