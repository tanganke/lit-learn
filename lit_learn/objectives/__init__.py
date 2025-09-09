"""
Objective functions for optimization.
"""

from lit_learn.core.objective import (
    BaseObjective,
    ObjectiveDict,
    ObjectiveList,
    OptimizationDirection,
)

from .adapters import *
from .classification import *
from .multi_objective import WeightedSumObjective
