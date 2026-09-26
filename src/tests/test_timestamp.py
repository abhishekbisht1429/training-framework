"""Tests for the timestamp that names checkpoint files."""

import re

from training_framework.util import timestamp_str


def test_timestamp_str_has_expected_shape():
    assert re.fullmatch(r"\d{8}_\d{6}_\d{9}", timestamp_str())
