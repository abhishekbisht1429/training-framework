"""Application-style entry point for spawned training integration tests."""

from training_framework.engine import Configurator, TrainingEngine


def main() -> None:
    with TrainingEngine(Configurator()) as engine:
        engine.start_session()


if __name__ == "__main__":
    main()
