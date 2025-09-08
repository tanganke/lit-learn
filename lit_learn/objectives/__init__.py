"""
Objective functions for optimization.
"""

from lit_learn.core.objective import (
    BaseObjective,
    ObjectiveDict,
    ObjectiveList,
    OptimizationDirection,
)

from .accuracy import MulticlassAccuracy
from .losses import LossAdapter
from .multi_objective import WeightedSumObjective
