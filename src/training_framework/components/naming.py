"""Component name syntax shared by the registry and the session."""


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
