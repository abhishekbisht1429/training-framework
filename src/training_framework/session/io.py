import os
from typing import TYPE_CHECKING, Any

import yaml

from training_framework.components import SessionHook
from training_framework.components import hook

if TYPE_CHECKING:
    from training_framework.session.base import Session


def write_session_config(session_dir: str, config: dict[str, Any]) -> None:
    os.makedirs(session_dir, exist_ok=True)
    config_dump_path = os.path.join(session_dir, "config.yaml")
    with open(config_dump_path, "w") as config_file:
        yaml.safe_dump(config, config_file)


@hook("config_dumper")
class ConfigDumper(SessionHook):
    def setup(self, session: "Session") -> None:
        write_session_config(
            session.session_config.session_dir,
            session.full_config,
        )

    def teardown(self, session: "Session") -> None:
        pass
