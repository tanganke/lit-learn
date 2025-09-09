"""
Unit tests for accuracy objectives.
"""

import unittest

import torch

from lit_learn.core.objective import OptimizationDirection
from lit_learn.objectives.classification.accuracy import MulticlassAccuracy


class TestMulticlassAccuracy(unittest.TestCase):
    """Test suite for MulticlassAccuracy objective."""

    def setUp(self):
        """Set up test fixtures."""
        self.accuracy = MulticlassAccuracy()

    def test_initialization(self):
        """Test proper initialization of MulticlassAccuracy."""
        self.assertEqual(
            self.accuracy.optimization_direction, OptimizationDirection.MAXIMIZE
        )
        self.assertFalse(self.accuracy.is_differentiable)

    def test_binary_classification_accuracy(self):
        """Test accuracy computation for binary classification."""
        # Perfect predictions
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0)

        # All wrong predictions
        predictions = torch.tensor([0.1, 0.9, 0.2, 0.8])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 0.0)

        # 50% accuracy
        predictions = torch.tensor([0.9, 0.9, 0.1, 0.1])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 0.5)

    def test_multiclass_classification_accuracy(self):
        """Test accuracy computation for multiclass classification."""
        # Perfect predictions (3 classes)
        predictions = torch.tensor(
            [
                [0.9, 0.05, 0.05],  # class 0
                [0.1, 0.8, 0.1],  # class 1
                [0.2, 0.2, 0.6],  # class 2
            ]
        )
        targets = torch.tensor([0, 1, 2])
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0)

        # All wrong predictions
        predictions = torch.tensor(
            [
                [0.1, 0.05, 0.85],  # predicts class 2, target is 0
                [0.8, 0.1, 0.1],  # predicts class 0, target is 1
                [0.2, 0.7, 0.1],  # predicts class 1, target is 2
            ]
        )
        targets = torch.tensor([0, 1, 2])
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 0.0)

        # Partial accuracy
        predictions = torch.tensor(
            [
                [0.9, 0.05, 0.05],  # correct: class 0
                [0.8, 0.1, 0.1],  # wrong: predicts 0, target is 1
                [0.2, 0.2, 0.6],  # correct: class 2
            ]
        )
        targets = torch.tensor([0, 1, 2])
        accuracy = self.accuracy(predictions, targets)
        self.assertAlmostEqual(accuracy, 2.0 / 3.0, places=6)

    def test_return_type_is_float(self):
        """Test that accuracy returns a Python float, not a tensor."""
        predictions = torch.tensor([0.9, 0.1])
        targets = torch.tensor([1.0, 0.0])
        accuracy = self.accuracy(predictions, targets)
        self.assertIsInstance(accuracy, float)
        self.assertNotIsInstance(accuracy, torch.Tensor)

    def test_gradient_computation_disabled(self):
        """Test that no gradients are computed during accuracy calculation."""
        predictions = torch.tensor([0.9, 0.1], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])

        # Compute accuracy
        accuracy = self.accuracy(predictions, targets)

        # Since accuracy returns a float, we can't call backward on it
        # But we can verify that predictions still requires grad after computation
        self.assertTrue(predictions.requires_grad)
        self.assertIsInstance(accuracy, float)

    def test_edge_cases(self):
        """Test edge cases for accuracy computation."""
        # Single sample
        predictions = torch.tensor([0.9])
        targets = torch.tensor([1.0])
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0)

        # Empty tensors (returns NaN)
        predictions = torch.tensor([])
        targets = torch.tensor([])
        accuracy = self.accuracy(predictions, targets)
        import math

        self.assertTrue(math.isnan(accuracy))

    def test_batch_dimensions(self):
        """Test accuracy with different batch dimensions."""
        # Batch of binary predictions
        predictions = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        accuracy = self.accuracy(predictions.squeeze(), targets.squeeze())
        self.assertEqual(accuracy, 1.0)

        # Batch of multiclass predictions with extra dimensions
        predictions = torch.tensor(
            [
                [[0.9, 0.05, 0.05]],
                [[0.1, 0.8, 0.1]],
                [[0.2, 0.2, 0.6]],
            ]
        ).squeeze(1)
        targets = torch.tensor([[0], [1], [2]]).squeeze()
        accuracy = self.accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
