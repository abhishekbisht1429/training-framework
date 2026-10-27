# Self-Supervised Training with DINO

[← Docs](../README.md) · [Project README](../../README.md)

This page builds a complete DINO run -- a ViT-S/16 trained on ImageNet
without labels -- from the built-in components, one part of the
configuration at a time, then runs it, checks it on a small scale, resumes
it, and trains a linear probe on what it learned. The last section shows
what changes for BYOL.

It assumes [Building an iteration](03-building-an-iteration.md) and
[Configuration](04-configuration.md). Every key used here is listed in the
reference: [self-supervised training](../reference/self-supervised.md) for
the DINO pieces, [image datasets](../reference/datasets.md) for the views,
[optimization](../reference/optimization.md) for the optimizer chain.

## What DINO trains

DINO trains a **student** network to predict what a **teacher** network
outputs for a different crop of the same image. The teacher is not trained
by gradients: after every optimizer step it moves a little toward the
student (an exponential moving average). Both end in a head that maps an
image to a distribution over 65536 prototypes; the teacher's distribution is
centred (a running mean is subtracted) and sharpened (a low temperature), and
the loss is the cross-entropy between the teacher's distribution for one
crop and the student's for every other crop.

Each image is cut into 2 large *global* crops (224 px) and 8 small *local*
ones (96 px). The student sees all 10; the teacher sees the 2 global ones.

The pieces, and what each is in the framework:

| DINO | Framework |
|---|---|
| ViT backbone | `patch_transformer` and its blocks |
| CLS token -> head | `backbone_with_head` with `pool: cls`, and `projection_head` |
| Multi-crop forward | `multi_view` |
| Student and teacher | `ema_teacher`, the model DDP wraps |
| Teacher momentum update | `ema_update` |
| Centring, sharpening, loss | `dino_distillation_loss` |
| Crops and augmentations | the dataset's `transform: views` |
| AdamW, warmup + cosine lr, no decay on norms, frozen last layer, clipping | the `optimizer` chain |

The finished configuration is about 140 lines. The sections below build it in
order; [the whole file](#the-whole-configuration) is at the end.

## Before you start

Install the vision extra, which the datasets need:

```bash
pip install "training-framework[vision]"
```

Write the entry point, `train.py`, as for every run:

```python
from training_framework.engine import Configurator, TrainingEngine


def main() -> None:
    with TrainingEngine(Configurator()) as engine:
        engine.start_session()


if __name__ == "__main__":
    main()
```

The datasets are optional components, so the session must import them. With
no components of your own, name their package as the `components_package`;
with a package of your own, import
`training_framework.components.builtin.datasets` from it instead:

```yaml
sessions:
  - session_config:
      rng_seed: 0
      sessions_dir: ./runs/dino
      max_iterations: 125100      # 100 epochs of 1251 steps at batch 1024
      device: cuda
      components_package: training_framework.components.builtin.datasets
    ddp:
      world_size: 8
      backend: nccl
    role_bindings:
      model: ema_teacher
      dataset: imagenet
```

Everything that follows goes inside this `sessions[0]` entry. The model the
run trains is `ema_teacher` -- the student and teacher pair -- and the
dataset is `imagenet`.

## 1. The data: ten views of each image

```yaml
    imagenet:
      root: /data/imagenet
      split: train
      transform: views
      mean: imagenet
      std: imagenet
      image_dtype: uint8
      views:
        - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 1.0}
        - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 0.1, solarize_p: 0.2}
        - {count: 8, size: 96, scale: [0.05, 0.4], blur_p: 0.5}

    data_manager: {batch_size: 1024, num_workers: 10, pin_memory: true}
    load_batch: {fields: [pixels, labels], non_blocking: true}
    normalize_images: {input: pixels, output: views}
```

`transform: views` makes each sample a list of crops instead of one image,
as `views` lists them, global crops first. Every crop gets DINO's shared
augmentation -- random resized crop, flip, colour jitter, greyscale -- and the
blur and solarization its group sets: the first global crop is always
blurred, the second rarely blurred and sometimes solarized, the local crops
blurred half the time. Defaults and every key are in
[the views reference](../reference/datasets.md#views-for-self-supervised-training).

The data manager collates a batch into one batch per view, so `load_batch`
names the list of view batches `pixels` and the labels `labels` (DINO never
reads them).

`image_dtype: uint8` keeps the crops as bytes through the data loader's
shared memory and the copy to the GPU -- a quarter of the float size, which
matters with ten crops per image -- and `normalize_images` scales and
normalizes every view on the GPU, writing the list the networks read as
`views`.

`batch_size` is global: 8 ranks read 128 images each.

## 2. The student

The student is a ViT-S/16, its class token, and the DINO head. The ViT is
`patch_transformer` wired to three blocks, as in
[transformer blocks](../reference/transformer-blocks.md); every instance
here has the suffix `#student`, because the teacher will need its own:

```yaml
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
```

The learned positional table is 14 x 14 for the 224 px crops; for a 96 px
crop it is interpolated to 6 x 6, so one ViT encodes both sizes.

The head sits on the class token:

```yaml
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
```

`projection_head` is DINO's head: a 3-layer MLP to a 256-wide bottleneck,
L2-normalized, then a weight-normalized layer onto 65536 prototypes.
`backbone_with_head` runs the backbone, takes its class token (`pool: cls`)
and runs the head. It reads the tokens through the backbone's
`encode_with_mask`, the contract that says where the class token is, so it
cannot take the wrong token.

Last, the student must take all ten views in **one** call: DDP synchronises
gradients for one forward pass per backward. `multi_view` takes the list,
concatenates each run of same-sized views -- the 2 global, then the 8 local
-- and calls the network once per size:

```yaml
    multi_view#student:
      dependencies_role_bindings: {network: backbone_with_head#student}
```

## 3. The teacher

The teacher has the student's architecture. It is not a copy of the student
in memory -- a component's weights are saved by the component that created
them, so the teacher is the same blocks **configured again**, under
`#teacher`. YAML anchors (`&patch` ... `*patch`) make the two configurations
one, so they cannot drift apart:

```yaml
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
```

`views: 2` is the only difference: the teacher sees the two global crops.

`ema_teacher` holds the pair and is the `model`:

```yaml
    ema_teacher:
      dependencies_role_bindings:
        student: multi_view#student
        teacher: multi_view#teacher
```

When the session is built it checks that the two have the same parameter
names and shapes and share none, copies the student's weights into the
teacher, and freezes the teacher. Its `forward` is the student's, so DDP
trains the student only; `teacher_forward` runs the teacher without
gradients.

## 4. The iteration

```yaml
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
      teacher_temp: {start: 0.04, end: 0.07, kind: linear, steps: 37530}
    ema_update:
      momentum: {start: 0.996, end: 1.0, kind: cosine}
```

- `forward#student` calls the model DDP wraps -- the student -- on the ten
  views and writes the `(10 * B, 65536)` outputs as `student`.
- `forward#teacher` calls `teacher_forward` on the same list (the teacher
  keeps the first two) without gradients, and writes `teacher`.
- `dino_distillation_loss` reads both and writes `loss`: the teacher outputs
  are centred and sharpened at `teacher_temp`, which warms up from 0.04 to
  0.07 over the first 30 epochs (37530 steps) and holds there; the student's
  at 0.1. The view counts are stated, never guessed, and checked on every
  batch. Then the centre moves toward this batch's mean teacher output,
  averaged over all ranks.
- `ema_update` moves the teacher toward the student with a momentum that
  rises from 0.996 to 1 along a cosine over the run.

The steps run in the order their inputs and outputs imply; `backward` reads
`loss`, and `ema_update` declares that it follows `optimizer_step`:

```
forward_context (hook)  autocast, DDP no_sync
load_batch              -> pixels, labels
normalize_images        pixels -> views
forward#student         views -> student
forward#teacher         views -> teacher
dino_distillation_loss  student, teacher -> loss
backward                loss -> gradients
freeze_gradients        the prototype layer, during epoch 1
clip_gradients          total norm 3
optimizer_step          the student changes
ema_update              the teacher follows
```

The session prints this order (the execution graph) when it starts; keep
`show_execution_graph` on for a first run and check it.

The teacher follows the student *after* the step, and only when a step
actually happened: not inside a gradient-accumulation group, and not on a
step that fp16 skipped because its gradients overflowed. A checkpoint
written after an iteration therefore holds a matching pair.

## 5. Optimization

```yaml
    optimizer:
      optimizer: {name: AdamW, kwargs: {lr: 0.002, weight_decay: 0.04}}
      precision: bf16
      lr_scheduler:
        stages:
          - {name: LinearLR, kwargs: {start_factor: 0.001, total_iters: "$stage_iterations"}}
          - {name: CosineAnnealingLR, kwargs: {T_max: "$stage_iterations", eta_min: 1.0e-6}}
        milestones: [12510]
      param_groups:
        - {match: ["*.bias", "*norm*"], kwargs: {weight_decay: 0.0}}
    freeze_gradients:
      rules: [{match: "student.network.head.last_layer.*", until_iteration: 1251}]
    clip_gradients: {max_norm: 3.0}
```

- The lr is DINO's 0.0005 per 256 images, scaled to the batch of 1024. It
  warms up linearly over 10 epochs (12510 steps) and then follows a cosine
  to 1e-6.
- Biases and normalization weights get no weight decay.
- The prototype layer is kept fixed for the first epoch, which stabilizes
  the start of training. Parameter names follow the module tree:
  `student.network.head.last_layer.*` is the student's prototype layer.
- The total gradient norm is clipped at 3.
- `bf16` runs the forward passes under autocast. `fp16` works too; its
  skipped steps are handled as described above.

The teacher's parameters are in the optimizer too, but they never get a
gradient, so it never changes them.

Logging and checkpoints, once per epoch:

```yaml
    logger: {log_every: 50}
    checkpointer: {checkpoint_every: 1251}
```

## 6. Watch it train

The `logger` prints the learning rates, gradient norm and loss every
`log_every` iterations. For curves, add TensorBoard: the `tensorboard`
resource owns the writer, and `tensorboard_summaries` writes the values you
name to it.

```yaml
    tensorboard: {serve: false}         # or a server: {host: 0.0.0.0, port: 6006}
    tensorboard_summaries:
      call_every: 50
      scalars:
        train/loss: loss
        dino: dino_metrics
        ema/momentum: ema_momentum
      optimizer: true
```

and let the two DINO steps publish what they compute, in their entries from
[step 4](#4-the-iteration):

```yaml
    dino_distillation_loss:
      metrics_key: dino_metrics         # teacher_temp, center_norm, teacher_entropy
      # ... the keys from step 4
    ema_update:
      momentum_key: ema_momentum
      # ... the keys from step 4
```

The summaries hook reads `loss`, `dino_metrics` and `ema_momentum` the way a
step reads its inputs, so a name no step writes is an error when the run
starts. A mapping such as `dino_metrics` becomes one plot per entry
(`dino/teacher_temp`, ...); `optimizer: true` adds the learning rate of each
parameter group, the gradient norm, the loss and the steps taken. Only rank 0
writes. The events are in `<session_dir>/Tensorboard_tensorboard/`; with
`serve: false`, view them with `tensorboard --logdir <sessions_dir>`.

What to look at:

- `dino/teacher_entropy` -- the entropy of the targets. It should fall from
  near `log(65536)` = 11.1 and settle well above 0. Stuck high means the
  teacher outputs are uniform; falling to 0 means one prototype took over:
  both are collapse.
- `train/loss` -- falls quickly at first, then slowly; it is not comparable
  across temperatures.
- `dino/teacher_temp` and `ema/momentum` -- the schedules, as configured.

## 7. Run it

```bash
python train.py --config dino.yaml
```

Before a 100-epoch run, run the same file on a small scale. On a machine
without GPUs, with a few images in `imagenet/train/<class>/`:

```bash
python train.py --config dino.yaml --override \
  'sessions[0].imagenet.root=./imagenet' \
  'sessions[0].session_config.sessions_dir=./runs/check' \
  'sessions[0].session_config.max_iterations=3' \
  'sessions[0].session_config.device=cpu' \
  'sessions[0].ddp.backend=gloo' \
  'sessions[0].ddp.world_size=2' \
  'sessions[0].data_manager.batch_size=4' \
  'sessions[0].data_manager.num_workers=0' \
  'sessions[0].data_manager.pin_memory=false' \
  'sessions[0].optimizer.precision=fp32' \
  'sessions[0].optimizer.lr_scheduler.milestones=[1]' \
  'sessions[0].logger.log_every=1' \
  'sessions[0].checkpointer.checkpoint_every=2'
```

The milestone has to fit in the shortened run. Every rank builds the full
ViT-S and both heads, so this checks the whole configuration, the wiring and
two-rank DDP in a few seconds:

```text
Iteration 1/3 | lr: 2.000e-06, 2.000e-06
Iteration 2/3 | lr: 2.000e-03, 2.000e-03 | grad_norm: 4.279e+00 | loss: 1.0873e+01
Iteration 3/3 | lr: 1.000e-03, 1.000e-03 | grad_norm: 2.735e+00 | loss: 1.0841e+01
```

Two learning rates: the two parameter groups. A line is printed before its
iteration, so its gradient norm and loss are the previous iteration's.

## 8. Resume and extend

A checkpoint holds the student, the teacher, the centre, the optimizer, and
the length of every schedule, so a resumed run continues exactly:

```bash
python train.py --resume-session ./runs/dino/session_<time>/checkpoints/<time>
```

To train longer than planned, extend the session
([checkpoints, resume, and extend](05-checkpoints-and-resume.md)). The lr,
momentum and temperature schedules keep the length they were built for and
hold their final values past it, rather than stretching or restarting.

## 9. Use what it learned: a linear probe

DINO is judged by how well a linear classifier on the frozen teacher's
class token does. Import the teacher's ViT from the run, freeze it, and
train one linear layer:

```yaml
    import_components:
      patch_transformer#teacher:
        checkpoint: ./runs/dino/session_<time>/checkpoints/<time>
    role_bindings:
      model: backbone_with_head
      dataset: imagenet
    backbone_with_head:
      pool: cls
      frozen: ["*"]
      dependencies_role_bindings:
        backbone: patch_transformer#imported_teacher
        head: projection_head#probe
    projection_head#probe: {in_dim: 384, out_dim: 1000, num_layers: 1}
    imagenet: {root: /data/imagenet, split: train, transform: train, mean: imagenet, std: imagenet}
    load_batch: {fields: [images, labels]}
    forward: {args: [images], outputs: logits}
    compute#loss: {function: cross_entropy, args: [logits, labels], outputs: loss}
    optimizer:
      optimizer: {name: SGD, kwargs: {lr: 0.004, momentum: 0.9}}
```

Importing `patch_transformer#teacher` brings its three blocks with it, all
renamed `#imported_teacher` ([importing components](../reference/import-components.md)).
`frozen: ["*"]` freezes the whole backbone and keeps it in eval mode; a
`projection_head` with one layer is a linear layer. Only the head trains.

## BYOL instead

BYOL uses the same parts with three differences:

- Two views of each image (`views` with two groups of 224 px crops, one
  always blurred, one rarely blurred and sometimes solarized).
- The online (student) network is encoder, projector and **predictor**; the
  target (teacher) has no predictor. Nest two `backbone_with_head`s for the
  online side and tell `ema_teacher` which parts mirror each other:
  `student_part: network.backbone`, `teacher_part: network`.
- No centre or temperatures: the loss is `byol_loss`, run by `compute`.

The full BYOL configuration is in
[the reference](../reference/self-supervised.md#byol).

## Differences from the published recipe

The run follows the DINO paper's ViT-S/16 recipe, with these differences:
the weight decay stays at 0.04 instead of rising to 0.4, gradients are
clipped by their total norm rather than parameter by parameter, and the
teacher temperature warms up per step rather than per epoch. See
[the reference](../reference/self-supervised.md#differences-from-the-published-recipes).

## The whole configuration

```yaml
sessions:
  - session_config:
      rng_seed: 0
      sessions_dir: ./runs/dino
      max_iterations: 125100
      device: cuda
      components_package: training_framework.components.builtin.datasets
    ddp:
      world_size: 8
      backend: nccl
    role_bindings:
      model: ema_teacher
      dataset: imagenet

    imagenet:
      root: /data/imagenet
      split: train
      transform: views
      mean: imagenet
      std: imagenet
      image_dtype: uint8
      views:
        - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 1.0}
        - {count: 1, size: 224, scale: [0.4, 1.0], blur_p: 0.1, solarize_p: 0.2}
        - {count: 8, size: 96, scale: [0.05, 0.4], blur_p: 0.5}
    data_manager: {batch_size: 1024, num_workers: 10, pin_memory: true}
    load_batch: {fields: [pixels, labels], non_blocking: true}
    normalize_images: {input: pixels, output: views}

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
      teacher_temp: {start: 0.04, end: 0.07, kind: linear, steps: 37530}
      metrics_key: dino_metrics
    ema_update:
      momentum: {start: 0.996, end: 1.0, kind: cosine}
      momentum_key: ema_momentum

    optimizer:
      optimizer: {name: AdamW, kwargs: {lr: 0.002, weight_decay: 0.04}}
      precision: bf16
      lr_scheduler:
        stages:
          - {name: LinearLR, kwargs: {start_factor: 0.001, total_iters: "$stage_iterations"}}
          - {name: CosineAnnealingLR, kwargs: {T_max: "$stage_iterations", eta_min: 1.0e-6}}
        milestones: [12510]
      param_groups:
        - {match: ["*.bias", "*norm*"], kwargs: {weight_decay: 0.0}}
    freeze_gradients:
      rules: [{match: "student.network.head.last_layer.*", until_iteration: 1251}]
    clip_gradients: {max_norm: 3.0}

    logger: {log_every: 50}
    checkpointer: {checkpoint_every: 1251}
    tensorboard: {serve: false}
    tensorboard_summaries:
      call_every: 50
      scalars:
        train/loss: loss
        dino: dino_metrics
        ema/momentum: ema_momentum
      optimizer: true
```

---

**See also:** [Self-supervised training](../reference/self-supervised.md) for
every key of the DINO and BYOL components.
