import argparse
from omegaconf import OmegaConf


class Configurator:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('config', type=str)
        self._parser.add_argument('--override', type=str, nargs='*', default=None)

    def generate_config(self, args):
        args = self._parser.parse_args()
        config = OmegaConf.load(args.config)

        if args.override is not None:
            cli_config = OmegaConf.from_dotlist(args.override)
            config = OmegaConf.merge(config, cli_config)

        return OmegaConf.to_container(config)
