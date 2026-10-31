"""Paths to the values of a nested config, as text: `hooks.optimizer.lr`,
`steps[0].name`, `resources["a.b"].c` (a key that is not a plain name is
quoted)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*\Z")
_TOKEN = re.compile(
    r"""\.?(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)"""
    r"""|\[(?P<index>[0-9]+)\]"""
    r"""|\[(?P<quoted>"(?:[^"\\]|\\.)*")\]"""
)


def format_path(segments: tuple[str | int, ...]) -> str:
    text = ""
    for segment in segments:
        if isinstance(segment, int):
            text += f"[{segment}]"
        elif _PLAIN_KEY.match(segment):
            text += f".{segment}" if text else segment
        else:
            text += f"[{json.dumps(segment)}]"
    return text


def parse_path(text: str) -> tuple[str | int, ...]:
    """The segments of a path written as `format_path` writes it."""
    if not isinstance(text, str) or not text:
        raise ValueError(f"a config path must be a non-empty string; got {text!r}")
    segments: list[str | int] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if (
                match is None
                or (position == 0 and text.startswith("."))
                or (position > 0 and match.group("key") and text[position] != ".")
        ):
            raise ValueError(
                f"cannot read config path {text!r} at character {position + 1}: "
                'write keys as a.b, list items as a[0], and other keys as a["x.y"]'
            )
        if match.group("key") is not None:
            segments.append(match.group("key"))
        elif match.group("index") is not None:
            segments.append(int(match.group("index")))
        else:
            segments.append(json.loads(match.group("quoted")))
        position = match.end()
    return tuple(segments)


def normalize_path(text: str) -> str:
    return format_path(parse_path(text))


def flatten(config: Any) -> dict[str, Any]:
    """Every leaf value of `config` by its path. An empty mapping or list is
    a leaf of its own, so it is compared like any other value."""
    leaves: dict[str, Any] = {}

    def visit(value: Any, segments: tuple[str | int, ...]) -> None:
        if isinstance(value, Mapping) and value:
            for key, item in value.items():
                visit(item, segments + (key if isinstance(key, str) else str(key),))
        elif isinstance(value, list) and value:
            for index, item in enumerate(value):
                visit(item, segments + (index,))
        else:
            leaves[format_path(segments)] = value

    if not isinstance(config, Mapping):
        raise ValueError(f"a config must be a mapping; got {type(config).__name__}")
    visit(config, ())
    return leaves


def covers(prefix: str, path: str) -> bool:
    """Whether `path` is `prefix` or a value inside it."""
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def canonical(value: Any) -> str:
    """Text equal for equal config values and different for different ones
    (`1` and `1.0` and `true` stay apart)."""
    return json.dumps(value, sort_keys=True, default=repr)
