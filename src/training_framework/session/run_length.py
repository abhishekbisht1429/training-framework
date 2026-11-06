from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from training_framework.components.epochs import EpochSource
from training_framework.session.config import SessionConfig, run_length_field

if TYPE_CHECKING:
    from training_framework.session.components import SessionComponents


#: The resource a session asks how many iterations its epochs take.
EPOCH_SOURCE_NAME = "data_manager"


def resolve_session_config(
        settings: Mapping[str, Any],
        *,
        session_dir: str,
        components: "SessionComponents",
) -> SessionConfig:
    """The `SessionConfig` of normalized `settings`, its length resolved.

    The one place a configured run length becomes `max_iterations`: building
    a session and extending one both come here, after the components exist,
    so a length given in epochs is counted by the data that will run them.
    """
    field = run_length_field(settings)
    epochs = None
    if field == "epochs":
        epochs = settings["epochs"]
        max_iterations = _epoch_source(components).iterations_for_epochs(epochs)
        if (
                not isinstance(max_iterations, int)
                or isinstance(max_iterations, bool)
                or max_iterations < 0
        ):
            raise TypeError(
                f"{EPOCH_SOURCE_NAME}.iterations_for_epochs({epochs}) "
                "returned "
                f"{max_iterations!r}; it must return a non-negative integer"
            )
    else:
        max_iterations = settings["max_iterations"]
    return SessionConfig(
        rng_seed=settings["rng_seed"],
        session_dir=session_dir,
        max_iterations=max_iterations,
        epochs=epochs,
    )


def _epoch_source(components: "SessionComponents") -> EpochSource:
    if not components.has_resource(EPOCH_SOURCE_NAME):
        raise ValueError(
            "session_config.epochs needs a "
            f"'{EPOCH_SOURCE_NAME}' resource to count the epochs, and this "
            "session has none; configure one or give "
            "session_config.max_iterations instead"
        )
    source = components.get_resource(EPOCH_SOURCE_NAME)
    if not isinstance(source, EpochSource):
        raise TypeError(
            f"session_config.epochs needs '{EPOCH_SOURCE_NAME}' to count "
            f"epochs, but {type(source).__name__} is not an EpochSource; "
            "subclass training_framework.components.EpochSource and "
            "implement iterations_for_epochs, or give "
            "session_config.max_iterations instead"
        )
    return source
