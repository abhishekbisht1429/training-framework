"""Uniform configuration parsing for components that opt in.

A component sets ``config_schema`` to a dataclass; its fields validate the
*shape* of the configuration -- which keys are required, which are accepted,
and what they are coerced to -- while ``__post_init__`` validates *meaning*,
such as a head count dividing an embedding width.

Deliberately small. Nested specs whose entries are arbitrary user-supplied
values stay hand-parsed by the component that understands them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, get_origin


def _plain(value: Any) -> Any:
    """Return `value` with OmegaConf containers converted to dict/list."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _coerce(value: Any, annotation: Any) -> Any:
    """Coerce a sequence to a tuple when the field is annotated as one."""
    if not isinstance(value, list):
        return value
    if isinstance(annotation, str):
        # `from __future__ import annotations` leaves the annotation a string.
        return tuple(value) if annotation.lstrip().startswith("tuple") else value
    if annotation is tuple or get_origin(annotation) is tuple:
        return tuple(value)
    return value


def _accepted_keys(schema: type) -> list[str]:
    return [field.name for field in fields(schema)]


def parse_component_config(
        component_class: type,
        config: Mapping | None,
) -> Any:
    """Return `config` parsed into the component's ``config_schema``.

    Raises ``TypeError`` or ``ValueError`` -- whichever the underlying failure
    is -- wrapped in the ``Invalid <name> config: ...`` shape the framework
    uses elsewhere.
    """
    schema = component_class.config_schema
    if schema is None or not is_dataclass(schema):
        raise TypeError(
            f"{_name(component_class)} config_schema must be a dataclass; "
            f"got {schema!r}"
        )

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError(f"{_name(component_class)} config must be a mapping")

    values = _plain(config)
    accepted = _accepted_keys(schema)

    unknown = sorted(set(values) - set(accepted))
    if unknown:
        raise ValueError(
            f"Invalid {_name(component_class)} config: unknown keys "
            f"{unknown}. Accepted keys: {accepted}"
        )

    missing = [
        field.name
        for field in fields(schema)
        if field.name not in values
        and field.default is MISSING
        and field.default_factory is MISSING  # type: ignore[misc]
    ]
    if missing:
        raise ValueError(
            f"Invalid {_name(component_class)} config: missing required keys "
            f"{sorted(missing)}"
        )

    annotations = {field.name: field.type for field in fields(schema)}
    arguments = {
        key: _coerce(value, annotations[key])
        for key, value in values.items()
    }

    try:
        return schema(**arguments)
    except (TypeError, ValueError) as error:
        raise type(error)(
            f"Invalid {_name(component_class)} config: {error}"
        ) from error


def _name(component_class: type) -> str:
    return getattr(component_class, "name", component_class.__name__)


def field_names(schema: type) -> tuple[str, ...]:
    """Return the configuration keys a schema accepts."""
    return tuple(_accepted_keys(schema))
