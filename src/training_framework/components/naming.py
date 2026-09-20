"""Component name syntax shared by the registry and the session.

Two kinds of name pass through here, and they are not interchangeable:

* a **registered name** identifies a component *class* (`@resource("logger")`).
  It may never contain the separator -- `validate_component_name` enforces
  that, so the spelling stays free for instance names.
* an **instance name** identifies one component *instance* within a session
  (`logger`, and later `logger#2`). It is parsed here, once, into the
  registered name that implements it plus an optional instance suffix.

Parsing lives in this module alone. Letting an encoded name travel inward and
calling ``split`` at each consumer would make every such call site depend on
the encoding never changing.
"""

import re


INSTANCE_SEPARATOR = "#"
"""Separator reserved for naming individual instances of one component.

A session currently holds at most one instance of each registered component,
so the separator carries no meaning yet. It is rejected everywhere a component
name is written down in order to keep the spelling available: a name already
in use could not be given a new meaning later without breaking it.
"""


def validate_component_name(name, *, kind: str = "Component name") -> None:
    """Reject `name` if it uses the reserved instance separator.

    Names that are not strings are left alone: each caller reports a wrong
    type in its own terms, and this check has nothing to say about one.
    """
    if not isinstance(name, str):
        return
    if INSTANCE_SEPARATOR in name:
        raise ValueError(
            f"{kind} '{name}' must not contain '{INSTANCE_SEPARATOR}': the "
            "character is reserved for future per-instance component names."
        )


_INSTANCE_SUFFIX_PATTERN = re.compile(r"\A[A-Za-z0-9_]+\Z")
"""Accepted instance suffixes.

Letters, digits and underscores, so both `logger#2` and `logger#validation`
work. A suffix that says what the instance is for reads better everywhere the
name is shown -- an execution graph line, a checkpoint key, an error -- than
an ordinal does.
"""


def is_instance_name(name: str) -> bool:
    """Return whether `name` carries an instance suffix."""
    return isinstance(name, str) and INSTANCE_SEPARATOR in name


def parse_instance_name(name: str) -> tuple[str, str | None]:
    """Split an instance name into its registered name and instance suffix.

    A name without the separator is the sole instance of its component, so it
    parses to `(name, None)`: the plain names every session uses today are
    instance names that happen to carry no suffix.
    """
    if not isinstance(name, str):
        raise TypeError("Component instance name must be a string")
    if INSTANCE_SEPARATOR not in name:
        if not name:
            raise ValueError("Component instance name must not be empty")
        return name, None

    implementation, _, suffix = name.partition(INSTANCE_SEPARATOR)
    if not implementation:
        raise ValueError(
            f"Component instance name '{name}' has no component name before "
            f"'{INSTANCE_SEPARATOR}'."
        )
    if not _INSTANCE_SUFFIX_PATTERN.match(suffix):
        raise ValueError(
            f"Component instance name '{name}' has an invalid instance "
            f"suffix '{suffix}': it must be one or more letters, digits or "
            "underscores."
        )
    return implementation, suffix


def format_instance_name(implementation: str, suffix: str | None) -> str:
    """Return the instance name for `implementation` and `suffix`."""
    if suffix is None:
        return implementation
    return f"{implementation}{INSTANCE_SEPARATOR}{suffix}"
