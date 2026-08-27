import pytest

from training_framework.components.registry import (
    _SESSION_COMPONENT_REGISTRIES,
    _SHARED_COMPONENT_REGISTRY,
)
from training_framework.session.registry import _SESSION_TYPE_REGISTRY


_DEFAULT_SHARED_COMPONENTS = dict(_SHARED_COMPONENT_REGISTRY)
_DEFAULT_SESSION_COMPONENTS = {
    session_type: dict(registry)
    for session_type, registry in _SESSION_COMPONENT_REGISTRIES.items()
}
_DEFAULT_SESSION_TYPES = dict(_SESSION_TYPE_REGISTRY)


@pytest.fixture(autouse=True)
def reset_registries():
    _SHARED_COMPONENT_REGISTRY.clear()
    _SHARED_COMPONENT_REGISTRY.update(_DEFAULT_SHARED_COMPONENTS)

    for session_type in tuple(_SESSION_COMPONENT_REGISTRIES):
        if session_type not in _DEFAULT_SESSION_COMPONENTS:
            del _SESSION_COMPONENT_REGISTRIES[session_type]
    for session_type, defaults in _DEFAULT_SESSION_COMPONENTS.items():
        registry = _SESSION_COMPONENT_REGISTRIES.setdefault(session_type, {})
        registry.clear()
        registry.update(defaults)

    _SESSION_TYPE_REGISTRY.clear()
    _SESSION_TYPE_REGISTRY.update(_DEFAULT_SESSION_TYPES)
