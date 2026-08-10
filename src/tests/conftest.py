import pytest

from training_framework.training_session import STEP_REGISTRY, HOOK_REGISTRY, RESOURCE_REGISTRY

# NOTE, in production the registries should not be altered manually.
# For tests the registry needs to be cleared for every test case hence the fixture below is needed.
@pytest.fixture(autouse=True)
def clear_registries():
    STEP_REGISTRY.clear()
    HOOK_REGISTRY.clear()
    RESOURCE_REGISTRY.clear()