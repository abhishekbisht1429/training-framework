from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from training_framework.util import CaptureInitMeta

if TYPE_CHECKING:
    from training_framework.training_session import TrainingSession


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


class Hook(ABC, metaclass=CaptureInitMeta):
    name: str
    id: str


class SessionHook(Hook, ABC):
    @abstractmethod
    def setup(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def teardown(self, session: "TrainingSession") -> None:
        pass


class IterationHook(Hook, ABC):
    call_every: int

    @abstractmethod
    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def post_iteration_callback(self, session: "TrainingSession") -> None:
        pass


class LifecycleHook(SessionHook, IterationHook, ABC):
    """Wrap callbacks around a training iteration."""


class Resource(ABC, metaclass=CaptureInitMeta):
    name: str
    id: str

    @abstractmethod
    def setup(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def teardown(self, session: "TrainingSession") -> None:
        pass


class Step(ABC, metaclass=CaptureInitMeta):
    name: str
    id: str

    @abstractmethod
    def run(self, session: "TrainingSession") -> None:
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
