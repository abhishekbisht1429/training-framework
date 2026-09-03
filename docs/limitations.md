# Current Behavior and Limitations

[← Documentation index](README.md) · [Project README](../README.md)

## Current behavior and limitations

1. **Configured sessions start as one concurrent batch.** Every `sessions[]` entry contributes worker wrappers to the run; the engine does not provide sequencing or dependency ordering between sessions.

2. **DDP is currently single-node oriented.** The process rank is used directly as the CUDA device index. There is no separate local-rank abstraction for multi-node execution.

3. **The built-in DDP resource requires a compatible model resource.** The `model` role must resolve to a module accepted by PyTorch DDP. Distributed forward passes should use `session.get_resource("ddp").wrapped_model` while the session is active.

4. **Secondary-rank component roots are opt-in.** Ranks greater than zero retain roots in `ddp.parallel_components`, their recursive dependencies and wrapping targets, plus the DDP resource.

5. **Component registration is global per interpreter.** Resource, hook, and step names share one namespace within each shared or session-specific scope. Duplicate names in one scope fail; a matching scoped component overrides a shared component with the same name. Test suites that reset registration must account for Python's module import cache before expecting decorators to run again.

6. **Spawn requires importable and serializable definitions.** Define worker targets and component classes at module scope. Constructor arguments, state returned by `get_state()`, and checkpointed session-context values must be serializable.

7. **Heartbeat detection happens between framework stages.** A single long-running component call can exceed the deadline without sending another heartbeat. Set `--heartbeat-timeout` above the longest expected uninterrupted setup, hook, step, or teardown operation.

8. **Graceful stopping occurs between iterations.** A worker already inside a component call will finish or block there until the join timeout causes termination.

9. **Checkpoint files are trusted-code artifacts.** The built-in loader uses unrestricted Python deserialization. Never load an untrusted checkpoint.

10. **Exact data-pipeline replay is application-dependent.** DataLoader prefetching can move a sampler's issued position ahead of consumed batches. Persist and restore committed batch progress when exact continuation is required.

11. **An unavailable CUDA device currently falls back to CPU.** Validate the final `session.device` in application code when silent fallback is undesirable.

12. **TensorBoard is an external process.** Starting it requires an available executable and port, and including it in DDP parallel components would start one server per retained rank.