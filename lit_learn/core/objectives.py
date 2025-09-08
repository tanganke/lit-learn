"""
Objective functions for multi-task and multi-objective optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

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
        super().__init__()

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


class ObjectiveDict(nn.ModuleDict):
    """
    A dictionary of objectives for multi-task learning and multi-objective optimization.

    Similar to nn.ModuleDict but specifically designed for BaseObjective instances.
    Provides convenient methods for multi-task objective computation and management.

    Example:
        >>> objectives = ObjectiveDict({
        ...     'classification': cross_entropy_loss(),
        ...     'regression': mse_loss(),
        ...     'contrastive': ContrastiveLossObjective()
        ... })
        >>>
        >>> # Compute all objectives
        >>> predictions = {'classification': cls_pred, 'regression': reg_pred, ...}
        >>> targets = {'classification': cls_target, 'regression': reg_target, ...}
        >>> losses = objectives(predictions, targets)
        >>>
        >>> # Compute specific objectives
        >>> cls_loss = objectives['classification'](cls_pred, cls_target)
    """

    def __init__(self, objectives: Optional[Mapping[str, BaseObjective]] = None):
        """
        Initialize ObjectiveDict.

        Args:
            objectives: Dictionary mapping task names to BaseObjective instances
        """
        super().__init__(objectives)

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        task_subset: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute objectives for all or a subset of tasks.

        Args:
            predictions: Dictionary mapping task names to predictions
            targets: Dictionary mapping task names to targets
            task_subset: Optional list of task names to compute. If None, compute all.

        Returns:
            Dictionary mapping task names to computed objective values

        Raises:
            KeyError: If a task in task_subset doesn't exist in objectives
            KeyError: If a task exists in objectives but not in predictions/targets
        """
        results = {}
        tasks_to_compute = task_subset if task_subset is not None else list(self.keys())

        for task_name in tasks_to_compute:
            if task_name not in self:
                raise KeyError(f"Task '{task_name}' not found in objectives")
            if task_name not in predictions:
                raise KeyError(f"Task '{task_name}' not found in predictions")
            if task_name not in targets:
                raise KeyError(f"Task '{task_name}' not found in targets")

            objective = self[task_name]
            results[task_name] = objective(predictions[task_name], targets[task_name])

        return results

    def items(self) -> Iterator[tuple[str, BaseObjective]]:
        """Iterator over (task_name, objective) pairs."""
        return super().items()

    def get_optimization_directions(self) -> Dict[str, str]:
        """Get optimization direction for each objective."""
        return {name: obj.optimization_direction for name, obj in self.items()}

    def get_differentiable_tasks(self) -> List[str]:
        """Get list of task names that have differentiable objectives."""
        return [name for name, obj in self.items() if obj.is_differentiable]

    def get_non_differentiable_tasks(self) -> List[str]:
        """Get list of task names that have non-differentiable objectives."""
        return [name for name, obj in self.items() if not obj.is_differentiable]
