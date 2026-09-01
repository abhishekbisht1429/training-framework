from collections.abc import Mapping

from training_framework.session.base import Session
from training_framework.session.config import SessionConfig, TRAINING_SESSION_TYPE
from training_framework.session.registry import register_session_type


@register_session_type(TRAINING_SESSION_TYPE)
class TrainingSession(Session):
    """Training workflow with training-specific defaults and extension support."""

    @classmethod
    def _default_component_configs(cls) -> Mapping[str, Mapping]:
        return {
            "logger": {"log_every": 10},
            "checkpointer": {"checkpoint_every": 100},
        }

    def __init__(self, config: dict):
        super().__init__(config)

    def update_max_iters(self, new_max_iters):
        self._session_config = SessionConfig(
            rng_seed=self.session_config.rng_seed,
            session_dir=self.session_config.session_dir,
            max_iterations=new_max_iters,
        )
