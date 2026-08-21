import pytest

from training_framework.training_session import (
    HOOK_REGISTRY,
    RESOURCE_REGISTRY,
    STEP_REGISTRY,
)

_DEFAULT_STEP_COMPONENTS = dict(STEP_REGISTRY)
_DEFAULT_HOOK_COMPONENTS = dict(HOOK_REGISTRY)
_DEFAULT_RESOURCE_COMPONENTS = dict(RESOURCE_REGISTRY)


# NOTE: In production, the registries should not be altered manually.
# Tests reset the registries to their built-in components before each test.
@pytest.fixture(autouse=True)
def reset_registries():
    STEP_REGISTRY.clear()
    STEP_REGISTRY.update(_DEFAULT_STEP_COMPONENTS)

    HOOK_REGISTRY.clear()
    HOOK_REGISTRY.update(_DEFAULT_HOOK_COMPONENTS)

    RESOURCE_REGISTRY.clear()
    RESOURCE_REGISTRY.update(_DEFAULT_RESOURCE_COMPONENTS)
