from __future__ import annotations


class HpcError(Exception):
    """An HPC batch operation could not proceed."""


class PlanError(HpcError):
    """A manifest, profile, or training config cannot be compiled.

    Carries every problem found, so one `validate` run reports them all.
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        if len(self.problems) == 1:
            message = self.problems[0]
        else:
            message = f"{len(self.problems)} problems:\n" + "\n".join(
                f"- {problem}" for problem in self.problems
            )
        super().__init__(message)


class Problems:
    """Collects problems found while compiling; raises them together."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def add(self, problem: str) -> None:
        self._items.append(problem)

    def __bool__(self) -> bool:
        return bool(self._items)

    def raise_if_any(self) -> None:
        if self._items:
            raise PlanError(self._items)
