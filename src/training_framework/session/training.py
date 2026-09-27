from collections.abc import Mapping, Sequence
from copy import deepcopy

from omegaconf import OmegaConf

from training_framework.components.config import (
    COMPONENT_GROUPS,
    DEPENDENCIES_BINDINGS_KEY,
    ROLE_BINDINGS_KEY,
)
from training_framework.session.components import read_session_config
from training_framework.session.base import Session
from training_framework.session.config import (
    SessionConfig,
    TRAINING_SESSION_TYPE,
    normalize_session_config,
)
from training_framework.session.registry import register_session_type


@register_session_type(TRAINING_SESSION_TYPE)
class TrainingSession(Session):
    """Training workflow with training-specific defaults and extension support."""

    @classmethod
    def _default_component_configs(cls) -> Mapping[str, Mapping]:
        return {
            "logger": {"log_every": 10},
            "checkpointer": {"checkpoint_every": 100},
        }

    def __init__(self, config: dict):
        super().__init__(config)

    def update_max_iters(self, new_max_iters):
        effective_session_settings = deepcopy(self._session_settings)
        effective_session_settings["max_iterations"] = new_max_iters
        effective_session_settings = normalize_session_config(
            effective_session_settings
        )
        self._session_settings = effective_session_settings
        self._config["session_config"] = deepcopy(effective_session_settings)
        self._session_config = SessionConfig(
            rng_seed=self.session_config.rng_seed,
            session_dir=self.session_config.session_dir,
            max_iterations=new_max_iters,
        )

    @staticmethod
    def _changed_paths(
            current,
            effective,
            prefix: tuple[str, ...] = (),
    ) -> set[tuple[str, ...]]:
        if isinstance(current, Mapping) and isinstance(effective, Mapping):
            changed = set()
            for key in current.keys() | effective.keys():
                key_path = (*prefix, str(key))
                if key not in current or key not in effective:
                    changed.add(key_path)
                else:
                    changed.update(TrainingSession._changed_paths(
                        current[key], effective[key], key_path
                    ))
            return changed
        return set() if current == effective else {prefix}

    def _fill_extended_entries(
            self,
            current_config: dict,
            update_config: Mapping,
    ) -> None:
        """Give each component an override names but the configuration does
        not list the configuration it was built with, where the override
        puts it, so only what the override changes counts as changed.

        An override that names a listed component somewhere other than where
        it is listed is refused: merged, it would list the component twice.
        """
        listed = read_session_config(
            current_config,
            session_type=self._session_type,
        )
        named: list[tuple[str | None, str]] = []
        for key, value in update_config.items():
            if key in COMPONENT_GROUPS:
                if isinstance(value, Mapping):
                    named.extend((key, name) for name in value)
            elif key != "session_config":
                named.append((None, key))

        for group, name in named:
            if name in listed.components:
                listed_group = listed.components[name].group
                if listed_group != group:
                    path = name if listed_group is None else f"{listed_group}.{name}"
                    raise ValueError(
                        f"'{name}' is listed {listed.where(name)}; override "
                        f"'{path}.<key>' instead"
                    )
                continue
            try:
                built_with = self._components.config_for_extension(name)
            except ValueError:
                continue
            if group is None:
                current_config[name] = built_with
            else:
                if current_config.get(group) is None:
                    current_config[group] = {}
                current_config[group][name] = built_with

    def apply_extension_overrides(self, overrides: Sequence[str]) -> None:
        if not overrides:
            raise ValueError("Session extension requires at least one override")

        update_config = OmegaConf.to_container(
            OmegaConf.from_dotlist(list(overrides)),
            resolve=False,
        )
        if not isinstance(update_config, Mapping):
            raise ValueError("Session extension overrides must form a mapping")

        current_config = deepcopy(self._config)
        self._fill_extended_entries(current_config, update_config)

        effective = OmegaConf.to_container(
            OmegaConf.merge(
                OmegaConf.create(current_config),
                OmegaConf.create(update_config),
            ),
            resolve=False,
        )
        if not isinstance(effective, Mapping):
            raise ValueError("Effective extension config must be a mapping")
        effective = dict(effective)
        # Compared flattened, so the layout the configuration uses -- flat
        # or grouped -- does not change which paths count as changed.
        effective_view = read_session_config(
            effective,
            session_type=self._session_type,
        )
        changed_paths = self._changed_paths(
            read_session_config(
                current_config,
                session_type=self._session_type,
            ).comparable(),
            effective_view.comparable(),
        )

        session_paths = {
            path for path in changed_paths if path[0] == "session_config"
        }
        unsupported_session_paths = session_paths - {
            ("session_config", "max_iterations")
        }
        if unsupported_session_paths:
            names = ", ".join(
                ".".join(path) for path in sorted(unsupported_session_paths)
            )
            raise ValueError(
                "Session extension does not allow changes to: " + names
            )

        reserved_names = {
            "aliases",
            "component_bindings",
            "components",
            "import_components",
            ROLE_BINDINGS_KEY,
            "session_kwargs",
            "session_type",
        }
        reserved_changes = {
            path for path in changed_paths
            if path[0] in reserved_names
            or (len(path) > 1 and path[1] == DEPENDENCIES_BINDINGS_KEY)
        }
        if reserved_changes:
            names = ", ".join(
                ".".join(path) for path in sorted(reserved_changes)
            )
            raise ValueError(
                "Session extension does not allow changes to: " + names
            )

        component_names = sorted({
            path[0] for path in changed_paths
            if path[0] != "session_config" and path[0] not in reserved_names
        })
        for name in component_names:
            component_paths = frozenset(
                path[1:] for path in changed_paths if path[0] == name
            )
            component_config = effective_view.components[name].config
            self._components.apply_extension_config(
                name,
                component_config,
                component_paths,
            )

        if session_paths:
            self.update_max_iters(
                effective["session_config"]["max_iterations"]
            )
            effective["session_config"] = deepcopy(self._session_settings)

        self._config = deepcopy(effective)
        self._extension_config_history_pending = True
