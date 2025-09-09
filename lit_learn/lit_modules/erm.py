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
        metrics_on_prog_bar: bool = True,
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        self.initial_optimizers = optimizers
        self.metrics_on_prog_bar = metrics_on_prog_bar

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
        self.metrics = nn.ModuleDict(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizers = self.initial_optimizers
        return optimizers

    def _forward_step(self, batch, stage: str):
        inputs, targets = batch
        predictions = self.model(inputs)

        # Compute objective (loss)
        if self.objective is not None:
            loss = self.objective(predictions, targets)
            # Log training loss
            self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        # Compute and log training metrics
        metrics = self.metrics[f"{stage}_metrics"]
        if len(metrics) > 0:
            step_results = cast(MetricCollection, metrics)(predictions, targets)
            self.log_dict(
                {f"{stage}/{k}": v for k, v in step_results.items()},
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True,
                prog_bar=self.metrics_on_prog_bar,
            )
        else:
            step_results = {}

        if self.objective is not None:
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
