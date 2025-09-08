"""
Scalarization methods for multi-objective optimization.

These functions create composite objectives from multiple objectives using different scalarization strategies.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from lit_learn.core.objectives import BaseObjective, ObjectiveDict, ObjectiveList
from lit_learn.objectives import WeightedSumObjective


def linear_scalarization(
    objectives: Union[ObjectiveList, ObjectiveDict],
    weight: Optional[Union[torch.Tensor, Any]] = None,
    requires_grad: bool = False,
    device=None,
    dtype=torch.float32,
) -> WeightedSumObjective:
    """
    Create a weighted sum scalarization of multiple objectives.

    Combines objectives using linear weights: f(x) = sum(w_i * f_i(x))

    Args:
        objectives: ObjectiveList or ObjectiveDict containing BaseObjective instances
        weight: 1D tensor of weights for each objective. If None, uses equal weights (1/N).
               Must be convertible to torch.Tensor if not already a tensor.
        requires_grad: Whether weights should be learnable parameters that can be optimized
        device: Device to place the weight tensor on (e.g., 'cpu', 'cuda')
        dtype: Data type for the weight tensor (default: torch.float32)

    Returns:
        WeightedSumObjective instance that computes the weighted combination of objectives
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    # Convert ObjectiveDict to list if needed to get count
    if isinstance(objectives, (ObjectiveDict, Dict)):
        num_objectives = len(objectives)
    else:
        num_objectives = len(objectives)

    ret_obj = WeightedSumObjective(
        objectives=objectives,
        requires_grad=requires_grad,
        **factory_kwargs,
    )

    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            try:
                weight = torch.as_tensor(weight, **factory_kwargs)
            except Exception as e:
                raise ValueError("Weights must be convertible to a torch.Tensor") from e

        assert (
            weight.size(0) == num_objectives
        ), "Number of weights must match number of objectives"
        assert weight.dim() == 1, "Weights must be a 1-dimensional tensor"

        ret_obj.weight.data = weight

    return ret_obj
