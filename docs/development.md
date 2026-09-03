# Development and Testing

[← Documentation index](README.md) · [Project README](../README.md)

## Testing

Install the package in editable mode and run:

```bash
python -m pip install -e .
python -m pytest
```

Run a focused file with:

```bash
python -m pytest src/tests/test_engine.py -q
```

The GitHub Actions workflow runs the suite on Python 3.12 and Python 3.13.

The current tests cover areas including:

- config-driven session execution;
- component registration and dependency validation;
- lifecycle and shared-context behavior;
- state and RNG restoration;
- checkpoint resume and extension;
- analysis session-type registration, model recovery, and spawned execution;
- real spawned-worker continuation and error propagation;
- rank-specific DDP session reconstruction; and
- heartbeat/process-monitoring behavior.

## Project layout

```text
training-framework/
├── docs/                 # Detailed Markdown documentation
├── .github/
│   └── workflows/
│       └── python-tests.yaml
├── src/
│   ├── training_framework/
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── config.py
│   │   │   ├── graph.py
│   │   │   ├── registry.py
│   │   │   └── builtin/
│   │   │       ├── __init__.py
│   │   │       ├── analysis.py
│   │   │       ├── checkpointing.py
│   │   │       ├── data.py
│   │   │       ├── distributed.py
│   │   │       ├── observability.py
│   │   │       └── optimization.py
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── core.py
│   │   │   ├── supervision.py
│   │   │   └── worker.py
│   │   ├── session/
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── base.py
│   │   │   ├── components.py
│   │   │   ├── config.py
│   │   │   ├── io.py
│   │   │   ├── runtime.py
│   │   │   ├── state.py
│   │   │   └── training.py
│   │   ├── dataloader.py
│   │   └── util.py
│   └── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```