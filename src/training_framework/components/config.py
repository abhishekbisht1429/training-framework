import warnings
from collections.abc import Mapping


COMMON_RESERVED_CONFIG_NAMES = frozenset({
    "aliases",
    "component_bindings",
    "components",
    "session_config",
    "session_kwargs",
    "session_type",
})


def component_bindings_from_config(config: Mapping) -> Mapping[str, str]:
    if "component_bindings" in config and "aliases" in config:
        raise ValueError(
            "Configure either 'component_bindings' or deprecated 'aliases', "
            "not both"
        )
    if "aliases" in config:
        warnings.warn(
            "The top-level 'aliases' entry is deprecated; use "
            "'component_bindings' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return config["aliases"]
    return config.get("component_bindings", {})


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


def reject_legacy_components_entry(config: Mapping) -> None:
    if "components" in config:
        raise ValueError(
            "The top-level 'components' entry is no longer supported. "
            "Activate root components with top-level mappings such as "
            "'component_name: {}'. Dependencies that do not define a custom "
            "constructor are activated automatically."
        )
