"""
Unit tests for loss objectives.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lit_learn.core.objective import OptimizationDirection
from lit_learn.objectives.losses import LossAdapter


class TestLossAdapter(unittest.TestCase):
    """Test suite for LossAdapter objective."""

    def test_initialization_with_class(self):
        """Test initialization with PyTorch loss class."""
        # Test with MSE loss
        loss_adapter = LossAdapter(nn.MSELoss)
        self.assertEqual(
            loss_adapter.optimization_direction, OptimizationDirection.MINIMIZE
        )
        self.assertTrue(loss_adapter.is_differentiable)
        self.assertIsInstance(loss_adapter.loss_fn, nn.MSELoss)

        # Test with CrossEntropy loss and kwargs
        loss_adapter = LossAdapter(nn.CrossEntropyLoss, reduction="sum")
        self.assertIsInstance(loss_adapter.loss_fn, nn.CrossEntropyLoss)

    def test_initialization_with_instance(self):
        """Test initialization with PyTorch loss instance."""
        mse_loss = nn.MSELoss(reduction="mean")
        loss_adapter = LossAdapter(mse_loss)
        self.assertEqual(
            loss_adapter.optimization_direction, OptimizationDirection.MINIMIZE
        )
        self.assertTrue(loss_adapter.is_differentiable)
        self.assertIs(loss_adapter.loss_fn, mse_loss)

    def test_initialization_with_function(self):
        """Test initialization with callable function."""
        loss_adapter = LossAdapter(F.mse_loss)
        self.assertEqual(
            loss_adapter.optimization_direction, OptimizationDirection.MINIMIZE
        )
        self.assertTrue(loss_adapter.is_differentiable)
        self.assertIs(loss_adapter.loss_fn, F.mse_loss)

    def test_custom_optimization_direction(self):
        """Test initialization with custom optimization direction."""
        loss_adapter = LossAdapter(
            nn.MSELoss, optimization_direction=OptimizationDirection.MAXIMIZE
        )
        self.assertEqual(
            loss_adapter.optimization_direction, OptimizationDirection.MAXIMIZE
        )

    def test_custom_differentiability(self):
        """Test initialization with custom differentiability setting."""
        loss_adapter = LossAdapter(nn.MSELoss, is_differentiable=False)
        self.assertFalse(loss_adapter.is_differentiable)

    def test_kwargs_with_callable_error(self):
        """Test that passing kwargs with callable raises error."""
        with self.assertRaises(AssertionError):
            LossAdapter(F.mse_loss, reduction="mean")

    def test_invalid_loss_fn_error(self):
        """Test error handling for invalid loss function."""
        with self.assertRaises(ValueError):
            LossAdapter("invalid_loss")

    def test_mse_loss_computation(self):
        """Test MSE loss computation."""
        loss_adapter = LossAdapter(nn.MSELoss)

        predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        targets = torch.tensor([1.5, 2.5, 2.5])

        loss_value = loss_adapter(predictions, targets)

        # Verify it's a tensor with gradient
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(loss_value.requires_grad)

        # Verify the computation is correct
        expected_loss = F.mse_loss(predictions, targets)
        self.assertAlmostEqual(loss_value.item(), expected_loss.item(), places=6)

    def test_cross_entropy_loss_computation(self):
        """Test CrossEntropy loss computation."""
        loss_adapter = LossAdapter(nn.CrossEntropyLoss)

        # 3 classes, 2 samples
        predictions = torch.tensor(
            [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]], requires_grad=True
        )
        targets = torch.tensor([0, 1])

        loss_value = loss_adapter(predictions, targets)

        # Verify it's a tensor with gradient
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(loss_value.requires_grad)

        # Verify the computation is correct
        expected_loss = F.cross_entropy(predictions, targets)
        self.assertAlmostEqual(loss_value.item(), expected_loss.item(), places=6)

    def test_l1_loss_computation(self):
        """Test L1 loss computation."""
        loss_adapter = LossAdapter(nn.L1Loss)

        predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        targets = torch.tensor([1.2, 1.8, 3.1])

        loss_value = loss_adapter(predictions, targets)

        # Verify the computation is correct
        expected_loss = F.l1_loss(predictions, targets)
        self.assertAlmostEqual(loss_value.item(), expected_loss.item(), places=6)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss adapter."""
        loss_adapter = LossAdapter(nn.MSELoss)

        predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        targets = torch.tensor([1.5, 2.5, 2.5])

        loss_value = loss_adapter(predictions, targets)
        loss_value.backward()

        # Check that gradients are computed
        self.assertIsNotNone(predictions.grad)
        self.assertTrue((predictions.grad != 0).any())

    def test_loss_with_reduction_parameters(self):
        """Test loss adapter with different reduction parameters."""
        # Test with sum reduction
        loss_adapter_sum = LossAdapter(nn.MSELoss, reduction="sum")

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 2.5])

        loss_sum = loss_adapter_sum(predictions, targets)
        expected_sum = F.mse_loss(predictions, targets, reduction="sum")
        self.assertAlmostEqual(loss_sum.item(), expected_sum.item(), places=6)

        # Test with mean reduction (default)
        loss_adapter_mean = LossAdapter(nn.MSELoss, reduction="mean")
        loss_mean = loss_adapter_mean(predictions, targets)
        expected_mean = F.mse_loss(predictions, targets, reduction="mean")
        self.assertAlmostEqual(loss_mean.item(), expected_mean.item(), places=6)

        # Verify sum is larger than mean for this case
        self.assertGreater(loss_sum.item(), loss_mean.item())

    def test_functional_loss_usage(self):
        """Test using functional losses (F.mse_loss, etc.)."""
        loss_adapter = LossAdapter(F.mse_loss)

        predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        targets = torch.tensor([1.5, 2.5, 2.5])

        loss_value = loss_adapter(predictions, targets)
        expected_loss = F.mse_loss(predictions, targets)

        self.assertAlmostEqual(loss_value.item(), expected_loss.item(), places=6)

    def test_batch_processing(self):
        """Test loss adapter with batched inputs."""
        loss_adapter = LossAdapter(nn.MSELoss)

        # Batch size 4, feature size 3
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)

        loss_value = loss_adapter(predictions, targets)

        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertEqual(loss_value.shape, ())  # Scalar loss
        self.assertTrue(loss_value.requires_grad)

    def test_complex_loss_shapes(self):
        """Test loss adapter with complex tensor shapes."""
        loss_adapter = LossAdapter(nn.MSELoss)

        # 2D predictions and targets
        predictions = torch.randn(2, 3, 4, requires_grad=True)
        targets = torch.randn(2, 3, 4)

        loss_value = loss_adapter(predictions, targets)
        expected_loss = F.mse_loss(predictions, targets)

        self.assertAlmostEqual(loss_value.item(), expected_loss.item(), places=6)


if __name__ == "__main__":
    unittest.main()
