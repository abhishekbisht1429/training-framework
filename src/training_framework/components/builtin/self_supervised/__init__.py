"""Self-supervised training from generic parts: DINO and BYOL.

- `projection_head`: the MLP on a backbone's pooled features (a DINO head
  with prototypes, a BYOL projector or predictor).
- `multi_view`: one network call per resolution over a list of views.
- `ema_teacher` + `ema_update`: a student trained through DDP and a teacher
  kept as its moving average.
- `dino_distillation_loss`: DINO's loss and its centre; `losses.byol_loss` is BYOL's,
  run by the `compute` step.
- `schedules`: values scheduled over optimizer steps (momentum, temperature).

The backbone is any module resource, e.g. `patch_transformer`, pooled by
`backbone_with_head`'s `pool`. See `docs/reference/self-supervised.md`.
"""

from training_framework.components.builtin.self_supervised.dino import DINOLoss
from training_framework.components.builtin.self_supervised.heads import (
    ProjectionHead,
)
from training_framework.components.builtin.self_supervised.losses import (
    byol_loss,
    cross_view_pairs,
    dino_cross_entropy,
)
from training_framework.components.builtin.self_supervised.multi_view import (
    MultiView,
)
from training_framework.components.builtin.self_supervised.schedules import (
    Schedule,
    parse_schedule,
)
from training_framework.components.builtin.self_supervised.teacher import (
    EMATeacher,
    EMAUpdate,
)

__all__ = [
    "DINOLoss",
    "EMATeacher",
    "EMAUpdate",
    "MultiView",
    "ProjectionHead",
    "Schedule",
    "byol_loss",
    "cross_view_pairs",
    "dino_cross_entropy",
    "parse_schedule",
]
