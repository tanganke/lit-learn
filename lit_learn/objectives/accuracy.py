"""
Metrics and non-differentiable objectives.
"""

from typing import Any

import torch

from lit_learn.core.objective import BaseObjective


class MulticlassAccuracy(BaseObjective):
    """
    Example of a non-differentiable objective that returns a float.

    Accuracy is not differentiable w.r.t. model parameters, so we return
    a simple float value instead of a tensor that would require gradients.
    """

    def __init__(self):
        super().__init__(minimize=False, is_differentiable=False)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute accuracy as a float.

        Since accuracy is not differentiable, we return a Python float
        instead of a tensor to avoid unnecessary gradient computation.
        """
        with torch.no_grad():  # Explicitly disable gradients
            if predictions.dim() > 1:
                # Multi-class case: take argmax
                pred_classes = torch.argmax(predictions, dim=-1)
            else:
                # Binary case: threshold at 0.5
                pred_classes = (predictions > 0.5).float()

            correct = (pred_classes == targets).float()
            accuracy = correct.mean().item()  # Convert to Python float

        return accuracy
