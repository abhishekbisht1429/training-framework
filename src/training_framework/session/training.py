from collections.abc import Mapping

from training_framework.session.base import Session
from training_framework.session.config import SessionConfig, SessionMode


class TrainingSession(Session):
    """Training workflow with training-specific defaults and extension support."""

    @classmethod
    def _session_mode(cls) -> SessionMode:
        return SessionMode.TRAINING

    @classmethod
    def _default_component_configs(cls) -> Mapping[str, Mapping]:
        return {
            "logger": {"log_every": 10},
            "checkpointer": {"checkpoint_every": 100},
        }

    def __init__(self, config: dict):
        super().__init__(config)
        self._init_args = {
            "args": (config,),
            "kwargs": {},
        }

    def update_max_iters(self, new_max_iters):
        self._session_config = SessionConfig(
            rng_seed=self.session_config.rng_seed,
            session_dir=self.session_config.session_dir,
            max_iterations=new_max_iters,
        )
