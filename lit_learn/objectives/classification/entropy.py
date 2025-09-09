import torch
from torch import Tensor

from lit_learn.core.objective import OptimizationDirection
from lit_learn.objectives.adapters.losses import LossAdapter


def entropy_loss(logits: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.
        eps (float): A small value to avoid log(0). Default is 1e-8.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    # Ensure the logits tensor has 2 dimensions
    assert (
        logits.dim() == 2
    ), f"Expected logits to have 2 dimensions, found {logits.dim()}, {logits.size()=}"

    # Compute the softmax probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute the entropy loss
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()


class Entropy(LossAdapter):
    """
    Entropy loss as a differentiable objective.

    This class wraps the entropy loss function in a LossAdapter to make it
    compatible with the lit-learn framework. It can be used as a differentiable
    objective for optimization.

    Example:
        ```python
        from lit_learn.objectives.classification import Entropy
        objective = Entropy()
        ```

    Args:
        eps (float): A small value to avoid log(0). Default is 1e-8.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(
            loss_fn=lambda logits, targets: entropy_loss(logits, eps=eps),
            optimization_direction=OptimizationDirection.MINIMIZE,
            is_differentiable=True,
        )
        self.eps = eps
