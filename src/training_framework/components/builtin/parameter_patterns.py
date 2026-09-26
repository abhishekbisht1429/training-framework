"""Selecting parameters by name with glob patterns.

One rule for every configuration that picks parameters -- optimizer groups,
gradient freezing, frozen fine-tuning layers: `fnmatch` patterns over the
names `named_parameters()` gives, and a pattern that selects nothing is an
error rather than a rule that silently applies to nothing.
"""

from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any


def parameter_patterns(value: Any, path: str) -> list[str]:
    """Return a list of parameter-name glob patterns, one string or several."""
    if isinstance(value, str):
        value = [value]
    if (
            not isinstance(value, Sequence)
            or isinstance(value, (bytes, Mapping))
            or not value
            or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(
            f"{path} must be a non-empty list of parameter name patterns"
        )
    return list(value)


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def check_patterns_match(
        patterns: Sequence[str],
        names: Sequence[str],
        path: str,
) -> None:
    """Reject a pattern that selects no parameter: it is almost always a typo,
    and a rule that silently applies to nothing gives a run that works and is
    quietly wrong."""
    unmatched = [
        pattern for pattern in patterns
        if not any(fnmatchcase(name, pattern) for name in names)
    ]
    if unmatched:
        preview = ", ".join(names[:8]) + (", ..." if len(names) > 8 else "")
        raise ValueError(
            f"{path} patterns {unmatched} match no parameter. Parameter "
            f"names look like: {preview}"
        )


__all__ = ["check_patterns_match", "matches_any", "parameter_patterns"]
