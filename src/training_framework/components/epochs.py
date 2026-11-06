from abc import ABC, abstractmethod


class EpochSource(ABC):
    """A resource that defines what one epoch of a session's data is.

    A session whose configuration gives its length as `epochs` asks the
    resource it knows as `data_manager` how many iterations that many epochs
    take, once, when the session is built; the answer becomes
    `session_config.max_iterations`. The resource must say so by subclassing
    this, so a data manager that cannot count epochs is refused rather than
    guessed at.
    """

    @abstractmethod
    def iterations_for_epochs(self, epochs: int) -> int:
        """How many iterations deliver `epochs` epochs of the data.

        Called before `setup`, so it may use only what construction gave
        the resource: its configuration and its prerequisites. Raise when
        the data cannot run that many epochs.
        """
        raise NotImplementedError
