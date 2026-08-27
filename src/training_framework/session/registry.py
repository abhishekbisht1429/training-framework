from collections.abc import Callable
from typing import TypeVar

from training_framework.session.config import normalize_session_type


SessionClass = TypeVar("SessionClass", bound=type)
_SESSION_TYPE_REGISTRY: dict[str, type] = {}


def register_session_type(
        session_type: str,
) -> Callable[[SessionClass], SessionClass]:
    normalized = normalize_session_type(session_type)

    def wrapper(cls: SessionClass) -> SessionClass:
        from training_framework.session.base import Session

        if not isinstance(cls, type) or not issubclass(cls, Session):
            raise TypeError(
                "@register_session_type can only be applied to Session subclasses"
            )
        if normalized in _SESSION_TYPE_REGISTRY:
            existing = _SESSION_TYPE_REGISTRY[normalized]
            raise ValueError(
                f"Session type '{normalized}' is already registered by "
                f"{existing.__name__}"
            )
        _SESSION_TYPE_REGISTRY[normalized] = cls
        cls._registered_session_type = normalized
        return cls

    return wrapper


def session_class_for_type(session_type: str) -> type:
    normalized = normalize_session_type(session_type)
    try:
        return _SESSION_TYPE_REGISTRY[normalized]
    except KeyError as error:
        raise ValueError(
            f"No Session subclass registered for session_type "
            f"'{normalized}'. Import the module that registers it first."
        ) from error
