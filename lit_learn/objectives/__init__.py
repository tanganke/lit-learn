"""
Objective functions for optimization.
"""

from lit_learn.core.objectives import BaseObjective, ObjectiveDict

from .accuracy import MulticlassAccuracy
from .losses import LossAdapter
