"""
Multi-objective specific objectives for scalarization and combination.
"""

from typing import Any, List, Union

import torch
import torch.nn as nn

from lit_learn.core.objectives import (
    BaseObjective,
    ObjectiveList,
)


class WeightedSumObjective(BaseObjective):
    """
    Weighted sum objective that combines multiple objectives with learnable weights.

    This objective computes: f(x) = sum(w_i * f_i(x)) where w_i are learnable weights.
    """

    def __init__(
        self,
        objectives: ObjectiveList,
        requires_grad: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Initialize weighted sum objective.

        Args:
            objectives: List or dict of BaseObjective instances to combine
            requires_grad: Whether weights should be learnable (default: True)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            optimization_direction=objectives.optimization_direction,
            is_differentiable=objectives.is_differentiable,
        )

        self.num_objectives = len(self.objectives)
        assert self.num_objectives > 0, "At least one objective is required"

        self.objectives = objectives
        self.weight = nn.Parameter(
            torch.empty(self.num_objectives, **factory_kwargs),
            requires_grad=requires_grad,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset weights to uniform distribution."""
        self.weight.fill_(1 / self.num_objectives)

    def forward(self, predictions: Any, targets: Any) -> torch.Tensor:
        """
        Compute weighted sum of objectives.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Weighted sum of objective values
        """
        results = self.objectives(predictions, targets)
        for obj_idx in range(len(self.objectives)):
            results[obj_idx] = results[obj_idx] * self.weight[obj_idx]
        return sum(results)
