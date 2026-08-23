import pytest

from training_framework.registry import _COMPONENT_REGISTRY

_DEFAULT_COMPONENTS = dict(_COMPONENT_REGISTRY)


# NOTE: The registry is private production infrastructure. Tests reset it to
# its built-in components before each test to isolate decorator side effects.
@pytest.fixture(autouse=True)
def reset_registries():
    _COMPONENT_REGISTRY.clear()
    _COMPONENT_REGISTRY.update(_DEFAULT_COMPONENTS)
