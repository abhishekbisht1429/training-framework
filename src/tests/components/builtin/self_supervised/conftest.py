"""Shared set-up for the self-supervised component tests: the synthetic
multi-view data and configurations of `integration_ssl_components`, and DDP
replaced by the in-process recording stand-in."""

from __future__ import annotations

import importlib
import sys

import pytest

from tests.test_utils import stub_process_group

SSL_COMPONENTS = "tests.integration.integration_ssl_components"


@pytest.fixture(autouse=True)
def ssl_components(monkeypatch):
    stub_process_group(monkeypatch)
    # The registries are reset before every test; registering again needs a
    # fresh import of the module that declares the test components.
    existing = sys.modules.get(SSL_COMPONENTS)
    if existing is None:
        return importlib.import_module(SSL_COMPONENTS)
    return importlib.reload(existing)
