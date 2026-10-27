# Self-Supervised Training

[← Docs](../README.md) · [Built-in components](builtin-components.md)

DINO and BYOL are built from generic parts, with any backbone; here it is
[`patch_transformer`](transformer-blocks.md). For a walkthrough that builds
a DINO run step by step, see the
[DINO guide](../guide/11-self-supervised-dino.md). Each part is useful on
its own:

| Component | Kind | Does |
|---|---|---|
| [`projection_head`](#projection_head) | Resource | An MLP head on pooled features: a DINO head (with prototypes), a BYOL projector or predictor |
| [`backbone_with_head`](fine-tuning.md#backbone_with_head) | Resource | `head(pool(backbone(x)))`; `pool: cls` takes the class token of `patch_transformer` |
| [`multi_view`](#multi_view) | Resource | Runs a list of view batches through one network, one call per resolution |
| [`ema_teacher`](#ema_teacher) | Resource | A student trained through DDP and a teacher kept as its moving average |
| [`ema_update`](#ema_update) | Step | Moves the teacher toward the student after each optimizer step |
| [`dino_distillation_loss`](#dino_distillation_loss) | Stateful step | DINO's loss, with the centre it keeps |
| `byol_loss` | Function | BYOL's loss, run by the [`compute`](generic-steps.md) step |
| [dataset `views`](datasets.md#views-for-self-supervised-training) | Dataset option | Multi-crop and two-view augmentation |

The rest is the framework's own: [`load_batch` and `forward`](generic-steps.md),
the [optimizer chain](optimization.md) (warmup and cosine lr, no weight decay
on biases and norms, freezing DINO's last layer, gradient clipping), and
[checkpoints](../guide/05-checkpoints-and-resume.md), which hold the teacher,
the centre and every schedule's length, so a resumed run continues exactly.

## How an iteration runs

```
load_batch -> forward#student -> forward#teacher -> dino_distillation_loss | compute#loss
  -> backward -> freeze_gradients -> clip_gradients -> optimizer_step -> ema_update
```

- `load_batch` reads the dataset's list of view batches (global views first).
- `forward#student` calls the model DDP wraps -- `ema_teacher`, whose
  forward is the student's -- on every view, in one call, so DDP synchronises
  the student's gradients. `multi_view` concatenates same-sized views and
  encodes them together.
- `forward#teacher` calls `ema_teacher.teacher_forward` with `no_grad: true`.
  The teacher's `multi_view` takes the leading `views` only (DINO: the two
  global crops).
- The loss reads both outputs; the optimizer chain updates the student.
- `ema_update` runs after `optimizer_step` (it declares it), so the teacher
  follows the student *after* this iteration's update, and a checkpoint
  written after the iteration holds a matching pair.

## The teacher is a second instance

The teacher is configured, not copied: the same blocks again under another
suffix (`#teacher`), bound to the `teacher` role. A component that declares
prerequisites cannot be copied privately, because every tensor is saved by
the component that created it ([`ModuleResource`](../concepts/module-resource.md)).
YAML anchors keep the two configurations from drifting apart, and
`ema_teacher` refuses a teacher whose parameter names or shapes differ from
the student's, or that shares any parameter with it.

When the session is built, `ema_teacher` copies the student's weights into
the teacher and freezes it (`requires_grad=False`): DDP and the optimizer
leave the teacher alone, and only `ema_update` changes it. On resume the
teacher's saved weights are loaded after construction, so a teacher that has
diverged from the student comes back as it was.

## DINO

ViT-S/16 with DINO's ImageNet recipe (batch 1024, 100 epochs of 1251
optimizer steps; 2 global and 8 local crops):

```yaml
role_bindings:
  model: ema_teacher
  dataset: imagenet

imagenet:
  root: /data/imagenet
  split: train
  transform: views
  mean: imagenet
  std: imagenet
  views:
    - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 1.0}
    - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 0.1, solarize_p: 0.2}
    - {count: 8, size: 96, scale: [0.05, 0.4], blur_p: 0.5}

# The student: ViT-S/16, pooled at the class token, with a DINO head.
conv_patch_embedding#student: &patch {in_channels: 3, patch_size: 16, embed_dim: 384}
learned_positional_embedding_2d#student: &positions {grid_size: 14, embed_dim: 384}
torch_transformer_encoder#student: &encoder
  embed_dim: 384
  num_heads: 6
  num_layers: 12
  dim_feedforward: 1536
  dropout: 0.0
  activation: gelu
  norm_first: true
  final_norm: true
patch_transformer#student:
  class_token: true
  dependencies_role_bindings:
    patch_embedding: conv_patch_embedding#student
    positional_embedding: learned_positional_embedding_2d#student
    sequence_encoder: torch_transformer_encoder#student
projection_head#student: &head
  in_dim: 384
  hidden_dim: 2048
  out_dim: 256
  num_layers: 3
  prototypes: {num: 65536}
backbone_with_head#student:
  pool: cls
  dependencies_role_bindings:
    backbone: patch_transformer#student
    head: projection_head#student
multi_view#student:
  dependencies_role_bindings: {network: backbone_with_head#student}

# The teacher: the same, configured again; it sees the two global crops.
conv_patch_embedding#teacher: *patch
learned_positional_embedding_2d#teacher: *positions
torch_transformer_encoder#teacher: *encoder
patch_transformer#teacher:
  class_token: true
  dependencies_role_bindings:
    patch_embedding: conv_patch_embedding#teacher
    positional_embedding: learned_positional_embedding_2d#teacher
    sequence_encoder: torch_transformer_encoder#teacher
projection_head#teacher: *head
backbone_with_head#teacher:
  pool: cls
  dependencies_role_bindings:
    backbone: patch_transformer#teacher
    head: projection_head#teacher
multi_view#teacher:
  views: 2
  dependencies_role_bindings: {network: backbone_with_head#teacher}

ema_teacher:
  dependencies_role_bindings:
    student: multi_view#student
    teacher: multi_view#teacher

data_manager: {batch_size: 1024, num_workers: 10, pin_memory: true}
load_batch: {fields: [views, labels]}
forward#student: {args: [views], outputs: student}
forward#teacher:
  method: teacher_forward
  args: [views]
  no_grad: true
  outputs: teacher
dino_distillation_loss:
  out_dim: 65536
  num_student_views: 10
  num_teacher_views: 2
  teacher_temp: {start: 0.04, end: 0.07, kind: linear, steps: 37530}  # 30 epochs
ema_update:
  momentum: {start: 0.996, end: 1.0, kind: cosine}
optimizer:
  optimizer: {name: AdamW, kwargs: {lr: 0.002, weight_decay: 0.04}}  # 0.0005 * 1024 / 256
  lr_scheduler:
    stages:
      - {name: LinearLR, kwargs: {start_factor: 0.001, total_iters: "$stage_iterations"}}
      - {name: CosineAnnealingLR, kwargs: {T_max: "$stage_iterations", eta_min: 1.0e-6}}
    milestones: [12510]                                               # 10 epochs
  param_groups:
    - {match: ["*.bias", "*norm*"], kwargs: {weight_decay: 0.0}}
freeze_gradients:
  rules: [{match: "student.network.head.last_layer.*", until_iteration: 1251}]  # epoch 1
clip_gradients: {max_norm: 3.0}
```

with `session_config.max_iterations: 125100`.

Parameter names follow the module tree: `student.network.backbone.*` is the
student's ViT, `student.network.head.*` its head, `teacher.*` the teacher.
The `freeze_gradients` rule above keeps DINO's prototype layer fixed for the
first epoch.

## BYOL

The online network is encoder, projector and predictor; the target is
encoder and projector only. Two `backbone_with_head`s nest for the online
side, and `ema_teacher` pairs the target with the part of the online network
it mirrors: `student_part: network.backbone` (the encoder and projector,
inside the online `multi_view`) and `teacher_part: network`.

```yaml
role_bindings:
  model: ema_teacher
  dataset: imagenet

imagenet:
  root: /data/imagenet
  split: train
  transform: views
  mean: imagenet
  std: imagenet
  views:
    - {count: 1, size: 224, scale: [0.08, 1.0], blur_p: 1.0}
    - {count: 1, size: 224, scale: [0.08, 1.0], blur_p: 0.1, solarize_p: 0.2}

conv_patch_embedding#online: &patch {in_channels: 3, patch_size: 16, embed_dim: 384}
learned_positional_embedding_2d#online: &positions {grid_size: 14, embed_dim: 384}
torch_transformer_encoder#online: &encoder
  embed_dim: 384
  num_heads: 6
  num_layers: 12
  dim_feedforward: 1536
  dropout: 0.0
  activation: gelu
  norm_first: true
  final_norm: true
patch_transformer#online:
  class_token: true
  dependencies_role_bindings:
    patch_embedding: conv_patch_embedding#online
    positional_embedding: learned_positional_embedding_2d#online
    sequence_encoder: torch_transformer_encoder#online
projection_head#online: &projector
  in_dim: 384
  hidden_dim: 4096
  out_dim: 256
  num_layers: 2
  activation: relu
  norm: batch
  init_std: null
projection_head#predictor:
  in_dim: 256
  hidden_dim: 4096
  out_dim: 256
  num_layers: 2
  activation: relu
  norm: batch
  init_std: null
backbone_with_head#online_projection:            # encoder + projector
  pool: cls
  dependencies_role_bindings:
    backbone: patch_transformer#online
    head: projection_head#online
backbone_with_head#online:                       # ... + predictor
  dependencies_role_bindings:
    backbone: backbone_with_head#online_projection
    head: projection_head#predictor
multi_view#online:
  dependencies_role_bindings: {network: backbone_with_head#online}

conv_patch_embedding#target: *patch
learned_positional_embedding_2d#target: *positions
torch_transformer_encoder#target: *encoder
patch_transformer#target:
  class_token: true
  dependencies_role_bindings:
    patch_embedding: conv_patch_embedding#target
    positional_embedding: learned_positional_embedding_2d#target
    sequence_encoder: torch_transformer_encoder#target
projection_head#target: *projector
backbone_with_head#target:
  pool: cls
  dependencies_role_bindings:
    backbone: patch_transformer#target
    head: projection_head#target
multi_view#target:
  dependencies_role_bindings: {network: backbone_with_head#target}

ema_teacher:
  student_part: network.backbone
  teacher_part: network
  dependencies_role_bindings:
    student: multi_view#online
    teacher: multi_view#target

data_manager: {batch_size: 1024, num_workers: 10, pin_memory: true}
load_batch: {fields: [views, labels]}
forward#student: {args: [views], outputs: student}
forward#teacher:
  method: teacher_forward
  args: [views]
  no_grad: true
  outputs: teacher
compute#loss:
  function: training_framework.components.builtin.self_supervised.losses.byol_loss
  args: [student, teacher]
  constants: {num_views: 2}
  outputs: loss
ema_update:
  momentum: {start: 0.996, end: 1.0, kind: cosine}
optimizer:
  optimizer: {name: AdamW, kwargs: {lr: 0.001, weight_decay: 0.05}}
  param_groups:
    - {match: ["*.bias", "*norm*"], kwargs: {weight_decay: 0.0}}
```

## `projection_head`

| Key | Default | Meaning |
|---|---|---|
| `in_dim` | required | Width of the features it takes: `(B, in_dim)` |
| `out_dim` | `256` | Width of the MLP's output (DINO's bottleneck) |
| `hidden_dim` | `2048` | Width of the hidden layers |
| `num_layers` | `3` | Linear layers; `1` is a single linear layer |
| `activation` | `gelu` | `relu`, `gelu`, `silu` or `tanh`, after every layer but the last |
| `norm` | `none` | `none`, `batch` or `layer`, before each activation |
| `prototypes` | `null` | `{num, norm_last_layer: true}`: DINO's last layer, below |
| `init_std` | `0.02` | Truncated-normal initialization of the MLP's linear layers, zero bias; `null` keeps PyTorch's default |

With `prototypes`, the MLP's output is L2-normalized and goes through a
weight-normalized `Linear(out_dim, num, bias=False)` -- the submodule
`last_layer`, parameters `last_layer.gain` and `last_layer.direction`
(`weight = gain * direction / |direction|`, row by row) -- whose gain is
fixed at 1 when `norm_last_layer` is true. The
output is then `(B, num)`. Registered for every session type.

It takes one vector per sample. Put it after a backbone that returns tokens
with [`backbone_with_head`](fine-tuning.md#backbone_with_head)'s `pool`.

## `multi_view`

| Key | Default | Meaning |
|---|---|---|
| `views` | all | Use only the first `views` views |

Requires the `network` role. Called with a list of `(B, ...)` view batches,
it concatenates each run of consecutive same-shaped views, calls the network
once per run, and returns the outputs concatenated in view order: `(V * B,
...)`, view `v` in rows `v*B` to `(v+1)*B`. One call per resolution keeps
the global and local crops of DINO to two network calls, and the whole
student to one call of the DDP-wrapped model. A network with BatchNorm (the
BYOL projector) therefore normalizes same-sized views together.

## `ema_teacher`

| Key | Default | Meaning |
|---|---|---|
| `student_part` | `""` | Attribute path of the part of the student the teacher mirrors; `""` is the whole student |
| `teacher_part` | `""` | Attribute path of the part of the teacher that mirrors it |

Requires the `student` and `teacher` roles; bind `model` to it. Its
`forward` is the student's; `teacher_forward` runs the teacher without
gradients; `update_teacher(momentum)` sets every mirrored teacher parameter
to `momentum * teacher + (1 - momentum) * student`. Buffers -- BatchNorm
statistics -- are copied once at construction and are the teacher's own
after that, kept up to date by its forward passes.

Refused when the session is built: the same instance for both roles, a
teacher that shares a parameter with the student, mirrored parts whose
parameter or buffer names or shapes differ (the error names the first), and
a part path that does not exist.

## `ema_update`

| Key | Default | Meaning |
|---|---|---|
| `momentum` | `{start: 0.996, end: 1.0, kind: cosine}` | A [schedule](#schedules) in [0, 1] |
| `momentum_key` | none | Also write the momentum in effect (the last update's) under this key, every iteration |

Requires `model` and `optimizer`, and runs after `optimizer_step`, only
when that step changed the weights (`optimizer.stepped`): not inside a
gradient-accumulation group, and not on a step `fp16` skipped because its
gradients overflowed -- the student did not move, so neither does the
teacher. The model must have an `update_teacher` method (`ema_teacher` has);
anything else is an error on the first update.

## `dino_distillation_loss`

| Key | Default | Meaning |
|---|---|---|
| `out_dim` | required | The head's output width; the centre's size |
| `num_student_views` | required | How many views the student's output holds |
| `num_teacher_views` | `2` | How many the teacher's holds: the student's first ones |
| `student_temp` | `0.1` | Student temperature |
| `teacher_temp` | `0.04` | Teacher temperature, a [schedule](#schedules) |
| `center_momentum` | `0.9` | Momentum of the centre's moving average |
| `student_key`, `teacher_key`, `loss_key` | `student`, `teacher`, `loss` | The context keys it reads and writes |
| `metrics_key` | none | Also write, under this key, `{teacher_temp, center_norm, teacher_entropy}` for [`tensorboard_summaries`](builtin-components.md#tensorboard_summaries) |

The loss is the cross-entropy between the centred, sharpened teacher
distribution of each teacher view and the student's distribution of each
*other* view, averaged over those pairs, computed in float32. The view counts
are checked against every batch, never inferred. After the loss, the centre
moves toward the batch mean of the teacher output -- the mean over every
rank's batch when `torch.distributed` is initialized. The centre and the
schedule's length are the step's state, so they are checkpointed.

`teacher_entropy` (with `metrics_key`) is the mean entropy of the teacher's
centred, sharpened distributions -- the targets. It is the usual signal of
collapse: near 0, one prototype dominates; near `log(out_dim)`, the targets
are uniform.

## `byol_loss`

`training_framework.components.builtin.self_supervised.losses.byol_loss(online,
target, *, num_views=2)`: `2 - 2 cos(p, z)` between the online prediction of
each view and the target projection of each other view, averaged over the
batch and summed over the pairs (both directions, as BYOL symmetrizes it).
The target gets no gradient. Run it with `compute`, as above.

## Schedules

A momentum or temperature is a number (constant) or a schedule over
optimizer steps:

```yaml
momentum: {start: 0.996, end: 1.0, kind: cosine}           # over the whole run
teacher_temp: {start: 0.04, end: 0.07, kind: linear, steps: 37530}
```

`kind` is `constant`, `linear` or `cosine`. The value is `start` at the first
optimizer step and reaches `end` after `steps` of them -- by default the
whole run, the optimizer steps in `max_iterations` -- then holds there. The
position is the optimizer steps actually taken
([`optimizer.completed_steps`](optimization.md#the-chain)), so a step `fp16`
skips does not advance a schedule. That
length is saved with the component, so an `--extend-session` holds the final
value rather than stretching the schedule, as the optimizer's lr schedule
does.

## Differences from the published recipes

These are deliberate and can be added later:

- No weight-decay schedule (DINO raises it from 0.04 to 0.4); the weight
  decay is constant.
- Gradient clipping is on the total norm; DINO clips each parameter's
  gradient norm separately.
- No SyncBatchNorm: BYOL's projector and predictor normalize each rank's
  batch, and their running statistics differ slightly between ranks.
- DINO's teacher temperature warms up per optimizer step, not per epoch.
- No LARS optimizer (BYOL's ResNet recipe); AdamW is the usual choice for a
  ViT.
- Evaluation (k-NN, linear probe) is not part of training: train a probe
  from the checkpoint with [`backbone_with_head`](fine-tuning.md#backbone_with_head)
  (`pool: cls`, the backbone frozen) and [`import_components`](import-components.md).
