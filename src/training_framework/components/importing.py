"""Resolving objects that configuration names by dotted path."""

import importlib
from typing import Any

from torch import nn


def import_object(
        dotted_path: Any,
        context: str,
        *,
        example: str = "torch.nn.Linear",
) -> Any:
    """Import the object a fully-qualified dotted path names.

    `context` names the configuration entry the path came from, so every
    error says which line to fix.
    """
    if not isinstance(dotted_path, str) or "." not in dotted_path:
        raise ValueError(
            f"{context} must be a fully-qualified dotted path (e.g. "
            f"'{example}'); got {dotted_path!r}"
        )
    module_path, _, attr_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise ImportError(
            f"{context} {dotted_path!r} could not be imported: {error}"
        ) from error
    try:
        return getattr(module, attr_name)
    except AttributeError as error:
        raise ValueError(
            f"{context} {dotted_path!r} has no attribute {attr_name!r} in "
            f"module {module_path!r}"
        ) from error


def import_module_class(
        dotted_path: Any,
        context: str,
        *,
        example: str = "torch.nn.Linear",
) -> type[nn.Module]:
    """Import an `nn.Module` subclass named by a dotted path."""
    resolved = import_object(dotted_path, context, example=example)
    if not isinstance(resolved, type) or not issubclass(resolved, nn.Module):
        raise TypeError(
            f"{context} {dotted_path!r} does not resolve to an nn.Module "
            "subclass"
        )
    return resolved


__all__ = ["import_module_class", "import_object"]
