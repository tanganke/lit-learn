"""
Loss functions and adapters for multi-task and multi-objective optimization.

This module provides adapters for PyTorch losses and specialized multi-task/multi-objective
loss functions that don't exist in PyTorch.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lit_learn.core.objective import BaseObjective


class LossAdapter(BaseObjective):
    """
    Adapter to wrap PyTorch loss functions as objectives.

    This allows using any PyTorch loss function within the lit-learn framework
    while maintaining the BaseObjective interface.
    """

    def __init__(
        self,
        loss_fn: Union[nn.Module, callable],
        minimize: bool = True,
        is_differentiable: bool = True,
        **loss_kwargs,
    ):
        """
        Initialize loss adapter.

        Args:
            loss_fn: PyTorch loss function (nn.MSELoss, nn.CrossEntropyLoss, etc.)
            minimize: Whether this objective should be minimized (default: True for losses)
            is_differentiable: Whether the objective is differentiable (default: True)
            **loss_kwargs: Additional arguments passed to loss function
        """
        super().__init__(minimize=minimize, is_differentiable=is_differentiable)

        # If it's a class, instantiate it with kwargs
        if isinstance(loss_fn, type) and issubclass(loss_fn, nn.Module):
            self.loss_fn = loss_fn(**loss_kwargs)
        # If it's already an instance or callable, use directly
        elif callable(loss_fn):
            assert not loss_kwargs, "Cannot pass kwargs when loss_fn is callable"
            self.loss_fn = loss_fn
        else:
            raise ValueError("loss_fn must be a nn.Module class or callable function")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss using the wrapped PyTorch loss function."""
        return self.loss_fn(predictions, targets)
