import pytest

from training_framework.components.registry import (
    _ANALYSIS_COMPONENT_REGISTRY,
    _COMPONENT_REGISTRY,
)

_DEFAULT_COMPONENTS = dict(_COMPONENT_REGISTRY)
_DEFAULT_ANALYSIS_COMPONENTS = dict(_ANALYSIS_COMPONENT_REGISTRY)


# NOTE: The registry is private production infrastructure. Tests reset it to
# its built-in components before each test to isolate decorator side effects.
@pytest.fixture(autouse=True)
def reset_registries():
    _COMPONENT_REGISTRY.clear()
    _COMPONENT_REGISTRY.update(_DEFAULT_COMPONENTS)
    _ANALYSIS_COMPONENT_REGISTRY.clear()
    _ANALYSIS_COMPONENT_REGISTRY.update(_DEFAULT_ANALYSIS_COMPONENTS)
