"""A resource base class for models assembled from other resources.

`ModuleResource` is both an `nn.Module` and a `StatefulResource`, so a model
can attach other registered resources as real submodules and keep those
references across a checkpoint or a process spawn. The wiring happens during
construction: components are built prerequisite-first on every path -- fresh
configuration, checkpoint restore, and worker start-up -- so by the time a
constructor runs, everything it declared already exists.

Each component owns its own parameters: a parent excludes everything reachable
from an attached child when it captures state, so every tensor is checkpointed
exactly once, by the component that created it.
"""

from abc import ABC
from collections.abc import Mapping
from copy import deepcopy
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn

from training_framework.components.base import (
    Component,
    ComponentDependencyError,
    StatefulResource,
)

if TYPE_CHECKING:
    from training_framework.session.base import Session


class ModuleResource(nn.Module, StatefulResource, ABC):
    """An `nn.Module` resource that may be composed of other resources.

    Subclasses create their own parameters in `__init__`, like any other
    PyTorch module, and list the resources they attach in `linked_modules`.
    Those children are attached by `super().__init__()`, before the subclass
    body runs, so a constructor may size its own weights from them.

    A constructed component is complete: it needs no `setup()` to be usable,
    which is what the `trained_model` analysis path relies on.
    """

    _STATE_VERSION = 1

    linked_modules: ClassVar[tuple[str, ...]] = ()
    """Names of resources attached as submodules under the same attribute."""

    def __init__(self, config: Mapping | None = None) -> None:
        nn.Module.__init__(self)
        if config is not None and not isinstance(config, Mapping):
            raise TypeError(f"{self._component_name()} config must be a mapping")
        self._config = deepcopy(dict(config or {}))
        self._linked_components: dict[str, str] = {}
        self._parse_config_schema(self._config)
        self.attach_dependencies()

    @property
    def config(self) -> dict[str, Any]:
        return deepcopy(self._config)

    @property
    def linked_components(self) -> dict[str, str]:
        """Return a copy of the attribute -> component name mapping."""
        return dict(self._linked_components)

    # -- dependencies -----------------------------------------------------

    def attach_dependencies(self) -> None:
        """Attach every entry of `linked_modules`.

        Runs from `ModuleResource.__init__`, so subclasses can use the
        attached children while creating their own weights. Override to
        attach a resource conditionally or under a different attribute name.
        """
        for name in type(self).linked_modules:
            self.attach_linked_module(name, self.get_dependency(name))

    def attach_linked_module(self, attribute: str, component: nn.Module) -> None:
        """Attach `component` as a submodule owned by another component."""
        if not isinstance(component, nn.Module):
            raise TypeError(
                f"{self._component_name()} requires '{attribute}' to be an "
                f"nn.Module resource; got {type(component).__name__}"
            )
        current = getattr(self, attribute, None)
        if current is component:
            return
        if current is not None:
            raise ComponentDependencyError(
                f"{self._component_name()} cannot attach '{attribute}' to a "
                f"different {type(component).__name__}; it is already attached"
            )
        setattr(self, attribute, component)
        self._linked_components[attribute] = getattr(
            type(component),
            "name",
            type(component).__name__,
        )

    # -- state ------------------------------------------------------------

    def _tensors_owned_by_children(self) -> set[int]:
        owned: set[int] = set()
        for attribute in self._linked_components:
            child = getattr(self, attribute)
            for tensor in chain(child.parameters(), child.buffers()):
                owned.add(id(tensor))
        return owned

    plain_module_api: ClassVar[tuple[str, ...]] = (
        "attach_dependencies",
        "get_state",
        "set_state",
        "rollback_setup",
        "setup",
        "teardown",
    )
    """Members a privately owned component may not override.

    A component held as a private submodule is used purely as an
    `nn.Module`: the session never sets it up and never asks it for state, so
    overriding any of these would silently do nothing. Extend this list when
    `ModuleResource` grows another member the session drives.
    """

    @classmethod
    def usable_as_plain_module(cls, component_class: type) -> bool:
        """Whether this component class may be owned as an ordinary submodule.

        True when the session drives nothing about it: it asks for no
        prerequisites and overrides none of `plain_module_api`, so holding one
        privately loses nothing.
        """
        if not isinstance(component_class, type) or not issubclass(
                component_class, ModuleResource,
        ):
            return False
        if component_class.linked_modules:
            # It expects the session to hand it prerequisites, which only
            # happens for a component the session itself constructed.
            return False
        return all(
            getattr(component_class, member) is getattr(ModuleResource, member)
            for member in cls.plain_module_api
        )

    def _check_child_ownership(self) -> None:
        # Walks the whole tree, not just direct children: a component nested
        # inside an nn.Sequential or ModuleList would otherwise escape the
        # check.
        attached = {
            id(getattr(self, attribute))
            for attribute in self._linked_components
        }

        def visit(module: nn.Module, prefix: str) -> None:
            for attribute, child in module.named_children():
                path = f"{prefix}{attribute}"
                if id(child) in attached:
                    # Owned by another component; its subtree is its own
                    # responsibility.
                    continue
                if isinstance(child, Component):
                    if not self.usable_as_plain_module(type(child)):
                        raise ComponentDependencyError(
                            f"{self._component_name()} holds component "
                            f"'{path}' as a private submodule, but "
                            f"{type(child).__name__} is driven by the "
                            "session: it declares prerequisites or overrides "
                            f"one of {list(self.plain_module_api)}, which "
                            "would never run here. Bind it to a role and "
                            "attach it with attach_linked_module() instead."
                        )
                    # Otherwise it is an ordinary module that happens to be a
                    # component class. Its weights are captured here, and
                    # SessionComponents.get_state() confirms nothing else
                    # captures them.
                visit(child, f"{path}.")

        visit(self, "")

    def captured_tensors(self) -> dict[str, Any]:
        """Return the live tensors this component checkpoints, by state key.

        The session cross-checks these across components so that every tensor
        is captured exactly once, whatever shape the module tree takes.
        """
        owned_elsewhere = self._tensors_owned_by_children()
        return {
            key: value
            for key, value in self.state_dict(keep_vars=True).items()
            if id(value) not in owned_elsewhere
        }

    def get_state(self) -> dict[str, Any]:
        self._check_child_ownership()
        return {
            "version": self._STATE_VERSION,
            "linked": dict(self._linked_components),
            "state_dict": {
                key: value.detach().clone()
                for key, value in self.captured_tensors().items()
            },
        }

    def set_state(self, state: Mapping[str, Any] | None) -> None:
        if state is None:
            return

        self._check_child_ownership()
        linked = dict(state.get("linked", {}))
        if linked != self._linked_components:
            raise ValueError(
                f"{self._component_name()} was checkpointed with linked "
                f"components {linked} but is now linked to "
                f"{self._linked_components}"
            )

        expected = set(self.captured_tensors())
        provided = set(state["state_dict"])
        unexpected = sorted(provided - expected)
        if unexpected:
            raise ValueError(
                f"{self._component_name()} state has keys it does not own: "
                f"{unexpected}"
            )
        missing = sorted(expected - provided)
        if missing:
            raise ValueError(
                f"{self._component_name()} state is missing keys: {missing}"
            )

        try:
            # In place, never `assign=True`: parameter identity must survive
            # so parents that already attached this component stay valid.
            self.load_state_dict(state["state_dict"], strict=False)
        except RuntimeError as error:
            raise ValueError(
                f"{self._component_name()} could not load its state: {error}"
            ) from error

    def __getstate__(self) -> Any:
        # nn.Module defines __getstate__/__setstate__, which would otherwise
        # shadow Stateful's versioned reconstruction envelope through the MRO.
        return StatefulResource.__getstate__(self)

    def __setstate__(self, state: Any) -> None:
        StatefulResource.__setstate__(self, state)

    # -- lifecycle --------------------------------------------------------

    def setup(self, session: "Session") -> None:
        self.to(session.device)

    def teardown(self, session: "Session") -> None:
        # Attachments are deliberately not torn down: a restored or torn-down
        # model must stay usable.
        pass
