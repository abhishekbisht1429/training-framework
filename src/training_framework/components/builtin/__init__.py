"""Built-in component implementations registered by the component package."""

from training_framework.components.builtin.checkpointing import Checkpointer
from training_framework.components.builtin.analysis import AnalysisLogger
from training_framework.components.builtin.computation import (
    AnalysisForward,
    Compute,
    Forward,
    LoadBatch,
)
from training_framework.components.builtin.data import (
    AnalysisDataManager,
    DataManager,
)
from training_framework.components.builtin.distributed import DDPResource
from training_framework.components.builtin.layer_inspection import (
    LayerCapture,
    LayerInspector,
)
from training_framework.components.builtin.model import TrainedModel
from training_framework.components.builtin.observability import (
    Logger,
    Tensorboard,
    Timer,
)
from training_framework.components.builtin.optimization import (
    Backward,
    ClipGradients,
    ForwardContext,
    FreezeGradients,
    GradientProcessor,
    OptimizerResource,
    OptimizerStep,
)
from training_framework.components.builtin.transformer import (
    PatchTransformer,
    PooledPatchTransformer,
)

__all__ = [
    "AnalysisDataManager",
    "AnalysisForward",
    "AnalysisLogger",
    "Backward",
    "Checkpointer",
    "ClipGradients",
    "Compute",
    "DataManager",
    "DDPResource",
    "Forward",
    "ForwardContext",
    "FreezeGradients",
    "GradientProcessor",
    "LayerCapture",
    "LayerInspector",
    "LoadBatch",
    "Logger",
    "OptimizerResource",
    "OptimizerStep",
    "PatchTransformer",
    "PooledPatchTransformer",
    "Tensorboard",
    "Timer",
    "TrainedModel",
]
