"""
empirical risk minimization
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import lightning as L
import lightning.pytorch.loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import MetricCollection

from lit_learn.core.objective import BaseObjective

logger = logging.getLogger(__name__)


class ERM_LitModule(L.LightningModule):
    """
    Empirical Risk Minimization module.
    """

    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection

    def __init__(
        self,
        model: nn.Module,
        objective: Union[Callable, "BaseObjective"],
        train_metrics: Optional[MetricCollection] = None,
        val_metrics: Optional[MetricCollection] = None,
        test_metrics: Optional[MetricCollection] = None,
        optim: DictConfig = None,
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        self.optim_cfg = optim

        # Initialize metric collections
        self.train_metrics = train_metrics or MetricCollection({})
        self.val_metrics = val_metrics or MetricCollection({})
        self.test_metrics = test_metrics or MetricCollection({})

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        if self.optim_cfg is None:
            # Default optimizer configuration
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer
        else:
            # Use Hydra to instantiate optimizer from config
            optimizer = instantiate(self.optim_cfg, params=self.parameters())
            return optimizer

    def training_step(self, batch, batch_idx: int):
        """
        Training step for empirical risk minimization.

        Args:
            batch: Training batch containing inputs and targets
            batch_idx: Index of the current batch

        Returns:
            Loss tensor for backpropagation
        """
        inputs, targets = batch
        predictions = self.model(inputs)

        # Compute objective (loss)
        loss = self.objective(predictions, targets)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Compute and log training metrics
        if len(self.train_metrics) > 0:
            train_metrics = self.train_metrics(predictions, targets)
            self.log_dict(
                {f"train_{k}": v for k, v in train_metrics.items()},
                on_step=False,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Validation step for empirical risk minimization.

        Args:
            batch: Validation batch containing inputs and targets
            batch_idx: Index of the current batch
        """
        inputs, targets = batch
        predictions = self.model(inputs)

        # Compute objective (loss)
        loss = self.objective(predictions, targets)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log validation metrics
        if len(self.val_metrics) > 0:
            val_metrics = self.val_metrics(predictions, targets)
            self.log_dict(
                {f"val_{k}": v for k, v in val_metrics.items()},
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        """
        Test step for empirical risk minimization.

        Args:
            batch: Test batch containing inputs and targets
            batch_idx: Index of the current batch
        """
        inputs, targets = batch
        predictions = self.model(inputs)

        # Compute objective (loss)
        loss = self.objective(predictions, targets)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Compute and log test metrics
        if len(self.test_metrics) > 0:
            test_metrics = self.test_metrics(predictions, targets)
            self.log_dict(
                {f"test_{k}": v for k, v in test_metrics.items()},
                on_step=False,
                on_epoch=True,
            )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Prediction step for inference.

        Args:
            batch: Input batch
            batch_idx: Index of the current batch
            dataloader_idx: Index of the current dataloader

        Returns:
            Model predictions
        """
        inputs, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        return self.model(inputs)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        return self.model(x)
