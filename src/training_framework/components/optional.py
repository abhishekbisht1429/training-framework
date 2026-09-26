"""Built-in components that register only when their package is imported.

Their packages need optional dependencies, so they are not imported with the
other built-ins. Naming one without importing its package fails as an
unregistered name; the diagnostics use this table to say which import is
missing. Kept free of imports so the lookup never needs the dependency.
A test checks it against what each package registers.
"""

from typing import NamedTuple


class OptionalPackage(NamedTuple):
    module: str
    extra: str


VISION_DATASETS = OptionalPackage(
    module="training_framework.components.builtin.datasets",
    extra="vision",
)

#: Component name -> the package that registers it.
OPTIONAL_COMPONENTS: dict[str, OptionalPackage] = {
    name: VISION_DATASETS
    for name in (
        "cifar10",
        "flowers102",
        "imagenet",
        "inaturalist",
        "stanford_cars",
    )
}
