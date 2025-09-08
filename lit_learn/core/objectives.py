"""
Objective functions for multi-task and multi-objective optimization.
"""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Union

import torch
import torch.nn as nn

_ObjectiveOutput = Union[torch.Tensor, float]


class OptimizationDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MIXED = "mixed"  # For multi-objective with both min and max objectives
    UNDEFINED = "undefined"  # For objectives where direction is not defined


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
    optimization_direction: OptimizationDirection = OptimizationDirection.UNDEFINED
    is_differentiable: bool = False  # Most objectives are differentiable

    def __init__(
        self,
        optimization_direction: OptimizationDirection = None,
        is_differentiable: bool = None,
    ):
        super().__init__()

        # Instance attributes override class attributes
        self.optimization_direction = (
            optimization_direction
            if optimization_direction is not None
            else self.__class__.optimization_direction
        )
        self.is_differentiable = (
            is_differentiable
            if is_differentiable is not None
            else self.__class__.is_differentiable
        )

    @abstractmethod
    def forward(self, predictions: Any, targets: Any) -> _ObjectiveOutput:
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


def _get_overall_optimization_direction(
    directions: Set[OptimizationDirection],
) -> OptimizationDirection:
    """Determine the overall optimization direction for a set of directions.

    Args:
        directions: Set of OptimizationDirection values

    Returns:
        Overall optimization direction (MINIMIZE, MAXIMIZE, MIXED, UNDEFINED)
    """
    if not directions:
        return OptimizationDirection.UNDEFINED

    if len(directions) == 1:
        return directions.pop()

    if directions == {
        OptimizationDirection.MINIMIZE,
        OptimizationDirection.MAXIMIZE,
    }:
        return OptimizationDirection.MIXED

    return OptimizationDirection.UNDEFINED


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
        predictions: Mapping[str, Any],
        targets: Mapping[str, Any],
        task_subset: Optional[List[str]] = None,
    ) -> Dict[str, _ObjectiveOutput]:
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

    def values(self) -> Iterator[BaseObjective]:
        """Iterator over objectives."""
        return super().values()

    def get_optimization_directions(self) -> Dict[str, OptimizationDirection]:
        """Get optimization direction for each objective."""
        return {name: obj.optimization_direction for name, obj in self.items()}

    def get_differentiable_objectives(self) -> Mapping[str, BaseObjective]:
        """Get mapping of task names to differentiable objectives."""
        return {name: obj for name, obj in self.items() if obj.is_differentiable}

    def get_non_differentiable_objectives(self) -> Mapping[str, BaseObjective]:
        """Get mapping of task names to non-differentiable objectives."""
        return {name: obj for name, obj in self.items() if not obj.is_differentiable}

    @property
    def is_differentiable(self) -> bool:
        """Whether all objectives are differentiable."""
        return all(obj.is_differentiable for obj in self.values())

    @property
    def optimization_direction(self) -> OptimizationDirection:
        return _get_overall_optimization_direction(
            {obj.optimization_direction for obj in self.values()}
        )


class ObjectiveList(nn.ModuleList):
    """
    A list of objectives for multi-objective optimization.

    Similar to nn.ModuleList but specifically designed for BaseObjective instances.
    Provides convenient methods for multi-objective computation where objectives
    don't have specific task names but are treated as a vector of objectives.

    Example:
        >>> objectives = ObjectiveList([
        ...     mse_loss(),
        ...     mae_loss(),
        ...     huber_loss()
        ... ])
        >>>
        >>> # Compute all objectives
        >>> losses = objectives(predictions, targets)  # Returns list of objective values
        >>>
        >>> # Access individual objectives
        >>> mse_value = objectives[0](predictions, targets)
    """

    def __init__(self, objectives: Optional[List[BaseObjective]] = None):
        """
        Initialize ObjectiveList.

        Args:
            objectives: List of BaseObjective instances
        """
        super().__init__(objectives)

    def forward(
        self,
        predictions: Iterator[Any],
        targets: Iterator[Any],
        task_subset: Optional[List[int]] = None,
    ) -> List[_ObjectiveOutput]:
        """
        Compute objectives for all or a subset of objectives.

        Args:
            predictions: Model predictions (same format for all objectives)
            targets: Target values (same format for all objectives)
            objective_indices: Optional list of indices to compute. If None, compute all.

        Returns:
            List of computed objective values in the same order as objectives

        Raises:
            IndexError: If an index in objective_indices is out of range
        """
        results = []
        indices_to_compute = (
            task_subset if task_subset is not None else range(len(self))
        )

        # Validate indices
        for idx in indices_to_compute:
            if idx >= len(self) or idx < -len(self):
                raise IndexError(
                    f"Objective index {idx} is out of range for {len(self)} objectives"
                )
        assert len(predictions) == len(indices_to_compute) and len(targets) == len(
            indices_to_compute
        ), (
            "Predictions and targets must match the number of objectives to compute. "
            f"Got {len(predictions)} predictions, {len(targets)} targets, and {len(indices_to_compute)} objectives to compute."
        )

        for idx in indices_to_compute:
            objective = self[idx]
            results.append(objective(predictions, targets))

        return results

    def __iter__(self) -> Iterator[BaseObjective]:
        """Iterator over objectives."""
        return super().__iter__()

    def get_optimization_directions(self) -> List[OptimizationDirection]:
        """Get optimization direction for each objective."""
        return [obj.optimization_direction for obj in self]

    def get_differentiable_objectives(self) -> List[BaseObjective]:
        """Get list of differentiable objectives."""
        return [obj for obj in self if obj.is_differentiable]

    def get_non_differentiable_objectives(self) -> List[BaseObjective]:
        """Get list of non-differentiable objectives."""
        return [obj for obj in self if not obj.is_differentiable]

    @property
    def is_differentiable(self) -> bool:
        """Whether all objectives are differentiable."""
        return all(obj.is_differentiable for obj in self)

    @property
    def optimization_direction(self) -> OptimizationDirection:
        return _get_overall_optimization_direction(
            {obj.optimization_direction for obj in self}
        )
