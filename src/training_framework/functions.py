"""Small functions for the built-in `compute` step to call by name.

Kept deliberately few: anything else comes from `torch` or a dotted path.
"""

from collections.abc import Mapping

import torch


def weighted_sum(
        *,
        weights: Mapping[str, float] | None = None,
        **terms: torch.Tensor,
) -> torch.Tensor:
    """Sum named terms, each scaled by its weight (default 1).

    Used to combine loss terms: `compute#loss` passes the terms as keyword
    arguments from the iteration context and the weights as a constant.
    """
    if not terms:
        raise ValueError("weighted_sum needs at least one term")
    weights = dict(weights or {})
    unknown = sorted(set(weights) - set(terms))
    if unknown:
        raise ValueError(
            f"weighted_sum has weights for {unknown}, which are not among "
            f"its terms {sorted(terms)}"
        )
    return sum(
        weights.get(name, 1.0) * value for name, value in terms.items()
    )


__all__ = ["weighted_sum"]
