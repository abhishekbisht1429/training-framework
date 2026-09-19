# API Summary

[← Documentation index](README.md) · [Project README](../README.md)

## API summary

The supported public imports are grouped by responsibility:

```python
from training_framework.components import (
    ModuleResource,
    Resource,
    Step,
    resource,
    step,
)
from training_framework.components.builtin import Checkpointer, TrainedModel
from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import (
    AnalysisSession,
    TrainingSession,
    register_session_type,
)
```

`Resource` defines `setup(session)`, `rollback_setup(session)`, and
`teardown(session)`. `SessionHook` defines `pre_session(session)`,
`rollback_pre_session(session)`, and `post_session(session)`.
The rollback methods are concrete no-ops by default, so existing subclasses
remain valid without implementing them. A component should override its
rollback method only when failed initialization can leave partial effects to
release.

`training_framework.dataloader` remains the public home of the infinite
samplers. `DistributedInfiniteSampler.get_state()` records how far an epoch
got across all ranks, and `set_state()` rebases a saved position onto the rank
and world size the sampler was built with, so a saved position can be restored
under a different topology.

`training_framework.engine` exports `LaunchTopology` and
`resolve_launch_topology`, which settle a run's process topology from the
command line, the environment and the configuration.

### `Configurator`

| Member | Purpose |
|---|---|
| `Configurator()` | Parse command-line operation and options |
| `mode` | `new`, `resume`, or `extend` |
| `session_configs` | Deep copy of parsed YAML session definitions in the new operation |
| `checkpoint_path` | Checkpoint path in resume or extend operations |
| `extension_overrides` | Session-relative dotlist overrides in the extend operation |
| `topology_overrides` | `ddp.world_size` / `master_addr` / `master_port` overrides, separated out because they describe the launch rather than the session |
| `new_max_iters` | Deprecated positional iteration limit, when supplied |
| `heartbeat_timeout` | Seconds a worker may go without marking progress |
| `stop_sync_grace_period` | Busy-poll duration before a pending DDP stop collective sleeps |
| `stop_sync_poll_interval` | Sleep duration between pending DDP stop-collective polls |
| `process_timeout_on_join` | Graceful process-join timeout |
| `debug` | Whether the parent only joins workers without monitoring them |
| `get_component_config(session_index, key)` | Return a deep copy of one component mapping, or `{}` for a listed no-config component |
| `get_all_component_configs(session_index)` | Return all selected component configs, excluding special entries |

`mode` controls the launch workflow. It is separate from `session_type`, which
selects the concrete session implementation.

### `TrainingEngine`

| Member | Purpose |
|---|---|
| `TrainingEngine(configurator)` | Create a process manager from CLI configuration |
| `start_session()` | Start all worker ranks for the active session; requires engine context |
| `register_session(config, *, session_type="training", session_kwargs=None)` | Construct a registered session type and its worker wrappers |
| `load_session(path, session_update_params=None)` | Load a checkpoint, optionally apply extension overrides, and prepare worker wrappers |
| `request_stop_all()` | Request cooperative shutdown of started workers |

Normal usage is:

```python
with TrainingEngine(Configurator()) as engine:
    engine.start_session()
```

The engine monitors workers while leaving the context.

### `Session`, `TrainingSession`, and `AnalysisSession`

| Member | Purpose |
|---|---|
| `Session` | Abstract base implementing the shared component and iteration lifecycle |
| `TrainingSession(config)` | Concrete training session with logger/checkpointer defaults and extension support |
| `AnalysisSession(config)` | Concrete analysis session with trained-model/logger defaults; configure the checkpoint under `trained_model.model_checkpoint_path` |
| `session_type` | Registered string identifying the concrete session workflow |
| `session_config` | Frozen `SessionConfig` containing seed, directory, and max iterations |
| `iteration` | Current completed/in-progress iteration counter |
| `device` | Active `torch.device` |
| `session_context` | Session-lifetime shared dictionary |
| `iteration_context` | Current-iteration shared dictionary; context-only |
| `send_heartbeat(stage)` | Mark worker progress with a stage label; call periodically inside long-running components (no-op outside a spawned worker) |
| `component_bindings` | Copy of the session's role-to-implementation bindings |
| `component_aliases` | Deprecated compatibility property for `component_bindings` |
| `resolve_component_name(name)` | Resolve an expected or actual component name to its registered name |
| `get_resource(name)` | Retrieve a configured resource |
| `has_resource(name)` | Test whether a resource is present |
| `get_all_resources()` | Return configured resources |
| `get_all_hooks()` | Return configured hooks |
| `get_all_steps()` | Return configured steps |
| `activate_component(name, config)` | Activate a registered component and its prerequisites; only before setup, and the way to add one that declares dependencies |
| `register_resource(resource)` | Add a registered resource instance that declares no dependencies |
| `register_hook(hook)` | Add a registered hook instance that declares no dependencies |
| `add_step(step)` | Add a registered step instance that declares no dependencies |
| `unregister_resource(name)` | Remove a resource from the session |
| `unregister_hook(name)` | Remove a hook from the session |
| `remove_step(name)` | Remove a step from the session |
| `get_state()` | Capture serializable session state |
| `set_state(state)` | Restore state into a session |
| `Session.from_state(state)` | Reconstruct and dispatch to the concrete session class recorded in state |
| `TrainingSession.update_max_iters(value)` | Replace a training session's maximum iteration count |

### Writing a component

| Member | Purpose |
|---|---|
| `Component.get_dependency(name)` | Return a declared prerequisite; valid only while the component is being constructed |
| `Component.has_dependency(name)` | Whether a declared prerequisite is active during construction |
| `Component.config_schema` | Optional dataclass; the configuration mapping is parsed into `self._cfg` |
| `parse_component_config(cls, config)` | Parse a mapping against a `config_schema` directly |
| `ExtendableComponent.apply_extension_config(config, changed_paths)` | Opt into configuration changes during `--extend-session` |
| `Stateful.get_state()` / `set_state(state)` | Capture and restore a component's own state |

Components are constructed prerequisite-first on every path, so
`get_dependency` is how a component takes hold of another one for good; see
[components](components.md#holding-another-component).

### `ModuleResource`

| Member | Purpose |
|---|---|
| `ModuleResource` | An `nn.Module` resource composed of other resources; see [built-in components](built-in-components.md#moduleresource) |
| `linked_modules` | Resource names attached as submodules under the same attribute |
| `attach_dependencies()` | Override to attach prerequisites conditionally or under another name |
| `attach_linked_module(attribute, component)` | Attach one component as a submodule owned by another component |
| `linked_components` | Copy of the attribute -> component name map; recorded in the checkpoint and checked on restore |
| `captured_tensors()` | The live tensors this component checkpoints, by state key |
| `usable_as_plain_module(cls)` | Whether a component class may be owned privately as an ordinary submodule |
| `plain_module_api` | Members a privately owned component may not override |

### Registration decorators

| API | Purpose |
|---|---|
| `@resource(name, session_type=None)` | Register a shared Resource, or scope it to one session type |
| `@hook(name, session_type=None)` | Register a shared Hook, or scope it to one session type |
| `@step(name, session_type=None)` | Register a shared Step, or scope it to one session type |
| `@requires_resource(name)` | Declare a resource prerequisite |
| `@requires_hook(name)` | Declare a Hook prerequisite for a Step |
| `@requires_step(name)` | Declare a Step prerequisite for a Step |
| `@wraps(name)` | Declare that a Hook wraps another Hook |
| `component_registry(session_type)` | Return shared components overlaid by the matching scoped registry |
| `topological_sort_of_components(..., session_type=...)` | Validate and order the selected session type's component graph |
| `@register_session_type(name)` | Register a concrete Session subclass for engine and checkpoint dispatch |
