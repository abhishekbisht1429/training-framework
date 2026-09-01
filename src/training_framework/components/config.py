from collections.abc import Mapping


COMMON_RESERVED_CONFIG_NAMES = frozenset({
    "aliases",
    "components",
    "session_config",
    "session_kwargs",
    "session_type",
})

SESSION_RESERVED_CONFIG_NAMES = {
    "analysis": frozenset({"model_checkpoint_path"}),
}


def reserved_config_names(session_type: str | None = None) -> frozenset[str]:
    if session_type is None:
        normalized_type = "training"
    elif not isinstance(session_type, str):
        raise TypeError("session_type must be a string or None")
    else:
        normalized_type = session_type.strip()
        if not normalized_type:
            raise ValueError("session_type must not be empty")

    return (
        COMMON_RESERVED_CONFIG_NAMES
        | SESSION_RESERVED_CONFIG_NAMES.get(normalized_type, frozenset())
    )


def selected_component_names(
        config: Mapping,
        *,
        session_type: str | None = None,
) -> list[str]:
    """Validate and return explicitly selected component names."""
    reserved_names = reserved_config_names(session_type)
    selected = config.get("components", [])
    if not isinstance(selected, list):
        raise TypeError("'components' must be a list of component names")

    names: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if not isinstance(name, str):
            raise TypeError("'components' entries must be strings")
        if not name:
            raise ValueError("'components' entries must not be empty")
        if name in reserved_names:
            raise ValueError(
                f"'{name}' is a reserved configuration name and cannot "
                "select a component"
            )
        if name in seen:
            raise ValueError(
                f"Component '{name}' appears more than once in 'components'"
            )
        seen.add(name)
        names.append(name)
    return names
