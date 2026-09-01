from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from training_framework.util import CaptureInitMeta, context_entry, context_exit

if TYPE_CHECKING:
    from training_framework.session.base import Session


class ComponentMeta(CaptureInitMeta):
    """Apply component lifecycle behavior to class-local overrides."""

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        if getattr(cls, "_context_managed_lifecycle", False):
            lifecycle_wrappers = {
                "setup": context_entry,
                "pre_session": context_entry,
                "teardown": context_exit,
                "post_session": context_exit,
            }
            for method_name, wrapper in lifecycle_wrappers.items():
                if method_name in namespace:
                    setattr(cls, method_name, wrapper(namespace[method_name]))

        return cls


class Component(ABC, metaclass=ComponentMeta):
    """Common base for every executable training-framework component."""

    name: str
    id: str
    _context_managed_lifecycle = False

    @classmethod
    @abstractmethod
    def _component_category_name(cls) -> str:
        """Return the top-level lifecycle category implemented by the class."""
        raise NotImplementedError


class Stateful(ABC):
    @abstractmethod
    def get_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: Any) -> None:
        raise NotImplementedError

    def __getstate__(self) -> Any:
        return self.get_state()

    def __setstate__(self, state: Any) -> None:
        self.set_state(state)


class Hook(Component, ABC):
    """Base category for session and iteration hooks."""

    _context_managed_lifecycle = True

    @classmethod
    def _component_category_name(cls) -> str:
        return "Hook"


class SessionHook(Hook, ABC):
    @abstractmethod
    def pre_session(self, session: "Session") -> None:
        pass

    @abstractmethod
    def post_session(self, session: "Session") -> None:
        pass


class IterationHook(Hook, ABC):
    call_every: int

    @abstractmethod
    def pre_iteration_callback(self, session: "Session") -> None:
        pass

    @abstractmethod
    def post_iteration_callback(self, session: "Session") -> None:
        pass


class LifecycleHook(SessionHook, IterationHook, ABC):
    """Wrap callbacks around a training iteration."""


class Resource(Component, ABC):

    _context_managed_lifecycle = True

    @classmethod
    def _component_category_name(cls) -> str:
        return "Resource"

    @abstractmethod
    def setup(self, session: "Session") -> None:
        pass

    @abstractmethod
    def teardown(self, session: "Session") -> None:
        pass


class Step(Component, ABC):

    @classmethod
    def _component_category_name(cls) -> str:
        return "Step"

    @abstractmethod
    def run(self, session: "Session") -> None:
        pass


class StatefulIterationHook(IterationHook, Stateful, ABC):
    pass


class StatefulSessionHook(SessionHook, Stateful, ABC):
    pass


class StatefulLifeCycleHook(LifecycleHook, Stateful, ABC):
    pass


StatefulLifecycleHook = StatefulLifeCycleHook


class StatefulStep(Step, Stateful, ABC):
    pass


class StatefulResource(Resource, Stateful, ABC):
    pass
