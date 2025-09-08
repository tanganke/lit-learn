"""
Objective functions for multi-task and multi-objective optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn


class BaseObjective(nn.Module, ABC):
    """
    Base class for optimization objectives.

    Focuses solely on computing objective values without any weighting.
    Weighting is handled by strategy classes (BaseScalarization, BaseWeightingStrategy).

    Attributes:
    - minimize: Whether this objective should be minimized (True) or maximized (False)
    - is_differentiable: Whether the objective is differentiable w.r.t. predictions
    - name: Human-readable name for logging and identification
    """

    # Class-level attributes that can be overridden by subclasses
    minimize: bool = True  # Most ML objectives are minimized (loss functions)
    is_differentiable: bool = True  # Most objectives are differentiable

    def __init__(
        self,
        minimize: bool = None,
        is_differentiable: bool = None,
    ):
        super().__init__(is_differentiable=is_differentiable)

        # Instance attributes override class attributes
        self.minimize = minimize if minimize is not None else self.__class__.minimize
        self.is_differentiable = (
            is_differentiable
            if is_differentiable is not None
            else self.__class__.is_differentiable
        )

    @abstractmethod
    def forward(self, predictions: Any, targets: Any) -> Union[torch.Tensor, float]:
        """
        Compute the objective value.

        Args:
            predictions: Model predictions (can be tensor, dict, tuple, list, etc.)
            targets: Target values (can be tensor, dict, tuple, list, etc.)

        Returns:
            Scalar tensor representing the objective value if differentiable,
            or a float if not differentiable (no gradient computation needed)
        """
        pass

    @property
    def optimization_direction(self) -> str:
        """Get optimization direction as string."""
        return "minimize" if self.minimize else "maximize"
