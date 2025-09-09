"""
empirical risk minimization
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

import lightning as L
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection

from lit_learn.core.objective import BaseObjective

logger = logging.getLogger(__name__)


class ERM_LitModule(L.LightningModule):
    """
    Empirical Risk Minimization module.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: Optional[OptimizerLRScheduler] = None,
        objective: Optional[Union[Callable, "BaseObjective"]] = None,
        metrics: Optional[MetricCollection] = None,
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        self.initial_optimizers = optimizers

        # Initialize metric collections
        if isinstance(metrics, dict):
            metrics = MetricCollection(metrics)

        train_metrics = (
            deepcopy(metrics) if metrics is not None else MetricCollection({})
        )
        val_metrics = deepcopy(metrics) if metrics is not None else MetricCollection({})
        test_metrics = (
            deepcopy(metrics) if metrics is not None else MetricCollection({})
        )
        self.metrics = MetricCollection(
            {"train": train_metrics, "val": val_metrics, "test": test_metrics}
        )

    def configure_optimizers(self):
        optimizers = self.initial_optimizers
        return optimizers

    def _forward_step(self, batch, stage: str):
        inputs, targets = batch
        predictions = self.model(inputs)

        # Compute objective (loss)
        loss = self.objective(predictions, targets)

        # Log training loss
        self.log(f"{stage}/loss", loss, prog_bar=True)

        # Compute and log training metrics
        if len(self.metrics[stage]) > 0:
            step_results = cast(MetricCollection, self.metrics[stage])(
                predictions, targets
            )
            self.log_dict(
                {f"{stage}/{k}": v for k, v in step_results.items()},
                on_step=False,
                on_epoch=True,
            )

        step_results["loss"] = loss
        return step_results

    def training_step(self, batch, batch_idx: int):
        step_results = self._forward_step(batch, stage="train")
        return step_results["loss"]

    def validation_step(self, batch, batch_idx: int):
        self._forward_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._forward_step(batch, stage="test")

    def forward(self, *args, **kwargs):
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        return self.model(*args, **kwargs)
