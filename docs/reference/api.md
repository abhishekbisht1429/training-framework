# API Summary

[← Docs](../README.md) · [Project README](../../README.md)

A flat list of the public imports and the members of each public type. This is
a lookup table, not an explanation — each entry links to the page that explains
it where one exists.

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
command line, the environment and the configuration, and `host_rendezvous`,
which binds the topology's rendezvous port (a free one when none is set) and
holds it in a `HostedRendezvous` until `close()`; the engine calls it for
every multi-process launch.

`training_framework.components.builtin` also exports the classes of the
[generic steps](generic-steps.md) (`LoadBatch`, `Forward`, `AnalysisForward`,
`Compute`) and of the [optimization chain](optimization.md)
(`OptimizerResource`, `ForwardContext`, `Backward`, `FreezeGradients`,
`ClipGradients`, `OptimizerStep`, `GradientProcessor`). They are configured by
name in YAML; only `GradientProcessor` is meant to be subclassed.

`training_framework.functions` holds the small functions `compute` finds by
name: `weighted_sum(*, weights=None, **terms)`; see
[`training_framework.functions`](generic-steps.md#training_frameworkfunctions).

`training_framework.components.naming` holds the instance-name syntax:
`INSTANCE_SEPARATOR`, `parse_instance_name(name)` (returning the component
name and the suffix), `format_instance_name`, `is_instance_name` and
`implementation_of`. It is the only module that parses the separator; see
[configuring a component more than once](../guide/02-wiring-components.md#configuring-a-component-more-than-once).

## `Configurator`

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

## `TrainingEngine`

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

## `Session`, `TrainingSession`, and `AnalysisSession`

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
| `get_all_resources()` | Return configured resources |
| `get_all_hooks()` | Return configured hooks |
| `get_all_steps()` | Return configured steps |
| `activate_component(name, config)` | Activate a registered component and its prerequisites; only before setup, and the way to add one that declares dependencies |
| `register_resource(resource)` | Add a registered resource instance built by hand; its declared prerequisites are given to it on registration, from `setup` onwards |
| `register_hook(hook)` | Add a registered hook instance built by hand; its declared prerequisites are given to it on registration, from `setup` onwards |
| `add_step(step)` | Add a registered step instance built by hand; its declared prerequisites are given to it on registration, from `setup` onwards |
| `unregister_resource(name)` | Remove a resource from the session |
| `unregister_hook(name)` | Remove a hook from the session |
| `remove_step(name)` | Remove a step from the session |
| `get_state()` | Capture serializable session state |
| `set_state(state)` | Restore state into a session |
| `Session.from_state(state)` | Reconstruct and dispatch to the concrete session class recorded in state |
| `TrainingSession.update_max_iters(value)` | Replace a training session's maximum iteration count |

## Writing a component

| Member | Purpose |
|---|---|
| `Component.get_dependency(name)` | Return a declared prerequisite, resolved for this component's own wiring and recorded; valid at any point in the component's life |
| `Component.has_dependency(name)` | Whether `name` is a declared prerequisite of this component |
| `Component.linked_components` | The asked name -> instance name map of prerequisites handed to this component |
| `Component.name` / `Component.id` | This instance's name (`logger#validation`) and its category-qualified id (`Hook.logger#validation`) |
| `Component.implementation_name` | The name this component's class was registered under, shared by every instance of it |
| `Component.instance_suffix` | The part after `#`, or `None` for the only instance of a component; use it to keep two instances' output apart |
| `Component.context_reads()` / `context_writes()` | The keys this instance reads / writes; defaults to the `@reads` / `@writes` declarations, overridden when keys come from configuration |
| `Component.config_schema` | Optional dataclass; the configuration mapping is parsed into `self._cfg` |
| `parse_component_config(cls, config)` | Parse a mapping against a `config_schema` directly |
| `ExtendableComponent.apply_extension_config(config, changed_paths)` | Opt into configuration changes during `--extend-session` |
| `Stateful.get_state()` / `set_state(state)` | Capture and restore a component's own state |
| `GradientProcessor.process(session, named_parameters)` | Base class (`training_framework.components.builtin`) for a step that edits gradients between `backward` and `optimizer_step`; see [custom gradient stages](optimization.md#custom-gradient-stages) |
| `Checkpointer.save_checkpoint(session, path)` | Write `session` as a checkpoint directory of plain data |
| `Checkpointer.load_checkpoint(path, map_location="cpu", restore_rng=True, *, on_mismatch="raise")` | Load a checkpointed session; `on_mismatch` (`"raise"`, `"reinit"` or instance names) decides what happens to a component whose saved state this version cannot take |
| `Checkpointer.load_component(path, name, *, with_dependencies=True, session_type=None)` | Rebuild one resource and the instances it was wired to, resolved through the checkpoint's own bindings, without adopting its RNG |
| `Checkpointer.load_component_state(path, name)` | Return one component's saved state without building anything |
| `Checkpointer.read_manifest(path)` | Return a checkpoint's `manifest.json` |
| `Component.state_version` / `migrate_state(from_version, state)` / `migrate_init_args(from_version, init_args)` | Version what a component checkpoints, and bring an older checkpoint forward |

`get_dependency` is the one way a component takes a prerequisite, whether in
`__init__` or at run time; see
[the component model](../concepts/component-model.md#taking-a-prerequisite).

**Removed:** `Session.get_resource` and `Session.has_resource` -- a component
takes its prerequisites with `get_dependency` (see
[the component model](../concepts/component-model.md#there-is-no-session-wide-lookup)).
`ComponentView`, `constructing_component` and `active_component_view` are no
longer exported from `training_framework.components`. Prerequisites are handed to a component
directly rather than through a bound view, so there is nothing left for them to
do. A test that built a component against a stub view can construct it and
then register it into a session, which gives it its prerequisites.

## `ModuleResource`

`ModuleResource` and its members — `get_dependency()`, `linked_components`,
`captured_tensors()`, `usable_as_plain_module()` and `plain_module_api` — are
documented in [`ModuleResource`](../concepts/module-resource.md#members).

## Registration decorators

| API | Purpose |
|---|---|
| `@resource(name, session_type=None)` | Register a shared Resource, or scope it to one session type |
| `@hook(name, session_type=None)` | Register a shared Hook, or scope it to one session type |
| `@step(name, session_type=None)` | Register a shared Step, or scope it to one session type |
| `@requires_resource(name)` | Declare a resource prerequisite |
| `@requires_hook(name)` | Declare a Hook prerequisite for a Step |
| `@requires_step(name)` | Declare a Step prerequisite for a Step |
| `@wraps(name)` | Declare that a Hook wraps another Hook |
| `@reads(*keys)` / `@writes(*keys)` | Declare the `iteration_context` keys a Step or IterationHook reads / writes; steps are [ordered and checked by them](../guide/02-wiring-components.md#ordering-by-dataflow) |
| `@activates(name)` | Declare a [companion](../guide/02-wiring-components.md#companions): activating this component activates `name` too, with no ordering or injection |
| `@rank_zero_only` | Declare that a distributed session builds this component on rank 0 only |
| `@singleton` | Declare that a session may hold only one instance of this component |
| `component_registry(session_type)` | Return shared components overlaid by the matching scoped registry |
| `topological_sort_of_components(..., session_type=...)` | Validate and order the selected session type's component graph |
| `@register_session_type(name)` | Register a concrete Session subclass for engine and checkpoint dispatch |

---

**See also:** [built-in components](builtin-components.md) for the components
these APIs register and configure.
