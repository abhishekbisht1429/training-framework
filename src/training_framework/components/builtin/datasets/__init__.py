"""Image datasets from torchvision, as resources for the ``dataset`` role.

Optional: needs ``pip install training-framework[vision]``, and is not
imported with the other built-ins. Import it from a module of your
``components_package`` so that every process registers it::

    import training_framework.components.builtin.datasets  # noqa: F401
"""

try:
    import torchvision  # noqa: F401
except ImportError as error:
    raise ImportError(
        "training_framework.components.builtin.datasets needs torchvision; "
        "install it with `pip install training-framework[vision]`"
    ) from error

from training_framework.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
)
from training_framework.components.builtin.datasets.cifar10 import CIFAR10
from training_framework.components.builtin.datasets.flowers102 import (
    Flowers102,
)
from training_framework.components.builtin.datasets.imagenet import ImageNet
from training_framework.components.builtin.datasets.inaturalist import (
    INaturalist,
)
from training_framework.components.builtin.datasets.stanford_cars import (
    StanfordCars,
)

__all__ = [
    "CIFAR10",
    "Flowers102",
    "ImageNet",
    "INaturalist",
    "StanfordCars",
    "TorchvisionDataset",
    "TorchvisionDatasetConfig",
]
