# Sessions

[← Documentation index](README.md) · [Project README](../README.md)

## Session lifecycle

Both `TrainingSession` and `AnalysisSession` are context managers and iterators.

```text
Construct session
    |
    +-- import component package
    +-- instantiate configured components
    |
Enter session
    +-- resource.setup() in dependency order
    +-- SessionHook.pre_session() in dependency order
    +-- create session directory and write config.yaml for rank 0
    |
Each iteration
    +-- selected IterationHook.pre_iteration_callback()
    +-- Step.run() in dependency order
    +-- selected IterationHook.post_iteration_callback() in reverse order
    +-- clear iteration_context
    |
Exit session
    +-- SessionHook.post_session() in reverse dependency order
    +-- resource.teardown() in reverse dependency order
    +-- clear session_context
```

The main phases are:

```text
NEW -> READY -> RUNNING -> FINISHED
                  |
                  +-> PAUSED when the context exits before max_iterations
```

If an iteration fails, its iteration counter is rolled back and its iteration context is cleared before the exception propagates.

## Shared contexts

### `iteration_context`

`session.iteration_context` is a dictionary for communication among hooks and steps during one iteration.

```python
session.iteration_context["batch"] = batch
loss = session.iteration_context["loss"]
```

It is:

- available only while the session context is active;
- visible to pre-hooks, steps, and post-hooks;
- cleared after every iteration; and
- not included in session checkpoints.

### `session_context`

`session.session_context` is a dictionary shared for the active session lifetime.

```python
session.session_context["best_loss"] = best_loss
```

It is included in `Session.get_state()` and restored with the concrete session. It is cleared when the session context exits. Values that exist when a checkpoint is created must therefore be serializable.

## Analysis sessions

Analysis sessions use the same resource, hook, step, context, and iteration
lifecycle as training sessions. Their effective registry combines shared
components with components registered using `session_type="analysis"`:

```python
from training_framework.components import Step, requires_resource, step


@step("report", session_type="analysis")
@requires_resource("trained_model")
class ReportStep(Step):
    def __init__(self, config):
        self.output_path = config["output_path"]

    def run(self, session):
        model = session.get_resource("trained_model").model
        # Analyze the model and write the configured report.
        ...
```

An analysis configuration uses the same `sessions` structure. The built-in
`trained_model` resource and analysis logger are default roots, so only
application analysis components need to be selected explicitly:

```yaml
sessions:
  - session_type: analysis
    model_checkpoint_path: ./runs/session_.../checkpoints/<checkpoint-name>

    session_config:
      rng_seed: 42
      sessions_dir: ./analysis-runs
      max_iterations: 1
      components_package: my_project.analysis_components
      device: cpu
      show_execution_graph: true

    report:
      output_path: ./analysis-runs/report.json
```

For direct execution, construct the concrete analysis subclass:

```python
from training_framework.session import AnalysisSession


analysis_config["model_checkpoint_path"] = (
    "./runs/session_.../checkpoints/<checkpoint-name>"
)
session = AnalysisSession(analysis_config)
```

Run the analysis entry through the same generic config path:

```bash
python -m my_project.train --config my_project/analysis.yaml
```

The checkpoint must contain a framework `TrainingSession`, not a standalone
model state dictionary. The source session must expose a model through the
`model` resource role, directly or through an alias, and that resource must
provide `to(device)` and `eval()`. During analysis setup, `trained_model` loads
the source session on CPU, moves the recovered model to the analysis device,
places it in evaluation mode, and exposes it through
`session.get_resource("trained_model").model`. Gradients remain enabled for
attribution-style analyses. Analysis does not activate the training
checkpointer by default.

Only load trusted checkpoints because session loading uses unrestricted Python
deserialization.

## Direct session execution

For single-process development or unit tests, either concrete session can be
driven without `TrainingEngine`. A training session can be run directly as:

```python
import yaml
from training_framework.session import TrainingSession


with open("my_project/config.yaml") as config_file:
    config = yaml.safe_load(config_file)["sessions"][0]

session = TrainingSession(config)

with session:
    for iteration in session:
        print(iteration)
```

Construct `AnalysisSession` in the same way shown in
[Analysis sessions](#analysis-sessions), then enter and iterate it with the same
pattern. Direct execution bypasses spawned-worker supervision, error pipes,
heartbeat monitoring, and rank-specific DDP reconstruction. Use
`TrainingEngine` for the normal managed execution path.