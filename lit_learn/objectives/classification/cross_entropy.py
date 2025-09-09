import torch.nn as nn

from lit_learn.objectives.adapters.losses import LossAdapter


class CrossEntropy(LossAdapter):
    """Cross-entropy loss objective for classification tasks."""

    def __init__(self, **loss_kwargs):

        super().__init__(nn.CrossEntropyLoss(**loss_kwargs))
