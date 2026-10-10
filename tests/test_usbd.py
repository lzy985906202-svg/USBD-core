"""Analytical and synthetic correctness fixtures; no benchmark configuration."""

import copy
import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from local_training import SGDLocalTrainer
from usbd import (DMDUnlearner, aggregate_and_subtract, flatten_params,
                  sample_count_weights, unlearn_from_snapshots)


class USBDTests(unittest.TestCase):
    def fit_linear(self, operator, initial):
        snapshots = [initial]
        for _ in range(6):
            snapshots.append(operator @ snapshots[-1])
        return DMDUnlearner(snapshots, 'cpu', singular_value_min=1e-12,
                            multiplier_limit=7.0)

    def test_known_linear_prediction(self):
        operator = torch.diag(torch.tensor([1.2, 0.7], dtype=torch.float64))
        initial = torch.tensor([2.0, -1.0], dtype=torch.float64)
        fit = self.fit_linear(operator, initial)
        expected = (torch.linalg.matrix_power(operator, 3) @ initial - initial).float()
        torch.testing.assert_close(fit.get_influence_vector(initial, 3), expected)

    def test_multiplier_cap_preserves_sign(self):
        operator = torch.diag(torch.tensor([-2.0, 3.0], dtype=torch.float64))
        initial = torch.tensor([1.0, 2.0], dtype=torch.float64)
        fit = self.fit_linear(operator, initial)
        # Cubed eigenvalues -8 and 27 are capped to -7 and 7.
        torch.testing.assert_close(fit.get_influence_vector(initial, 3),
                                   torch.tensor([-8.0, 12.0]))

    def test_zero_horizon_retains_projection_residual(self):
        snapshots = [torch.tensor([float(k), 0.0], dtype=torch.float64)
                     for k in (1, 2, 3)]
        fit = DMDUnlearner(snapshots, 'cpu', singular_value_min=1e-12,
                           multiplier_limit=7.0)
        torch.testing.assert_close(
            fit.get_influence_vector(torch.tensor([2.0, 3.0]), 0),
            torch.tensor([0.0, -3.0]))

    def test_sample_weighted_subtraction(self):
        weights = sample_count_weights({'a': 1, 'b': 3})
        self.assertEqual(weights, {'a': 0.25, 'b': 0.75})
        updated, delta = aggregate_and_subtract(
            torch.tensor([10.0, 20.0]),
            {'a': torch.tensor([2.0, 4.0]), 'b': torch.tensor([6.0, 8.0])},
            weights, damping=0.2)
        torch.testing.assert_close(delta, torch.tensor([5.0, 7.0]))
        torch.testing.assert_close(updated, torch.tensor([9.0, 18.6]))

    def test_input_model_and_bn_buffers_preserved(self):
        torch.manual_seed(19)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        original = copy.deepcopy(model.state_dict())
        theta = flatten_params(model)
        snapshots = {c: [theta.clone(), theta * scale, theta * scale ** 2]
                     for c, scale in [('a', 1.1), ('b', 0.9)]}
        result = unlearn_from_snapshots(
            model, snapshots, {'a': 0.25, 'b': 0.75}, horizon=1,
            damping=0.25, singular_value_min=1e-6,
            multiplier_limit=7.0, device='cpu')
        self.assertIsNot(result.model, model)
        self.assertGreater((flatten_params(result.model) - theta).norm().item(), 0)
        for key, value in original.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(value, dict(result.model.named_buffers())[key]))

    def test_mismatched_start_is_rejected(self):
        model = torch.nn.Linear(2, 2)
        theta = flatten_params(model)
        with self.assertRaisesRegex(ValueError, 'same global model'):
            unlearn_from_snapshots(
                model, {'a': [theta + 1, theta + 2]}, {'a': 1.0},
                horizon=1, damping=0.25, singular_value_min=1e-6,
                multiplier_limit=7.0, device='cpu')

    def test_local_sgd_momentum_persists(self):
        model = torch.nn.Linear(1, 2, bias=False).double()
        with torch.no_grad():
            model.weight.zero_()
        loader = DataLoader(TensorDataset(torch.ones(1, 1, dtype=torch.float64),
                                           torch.ones(1, dtype=torch.long)), batch_size=1)
        trainer = SGDLocalTrainer(loader, learning_rate=0.07, momentum=0.4,
                                  weight_decay=0.0, max_grad_norm=None, device='cpu')
        trainer(model)
        torch.testing.assert_close(model.weight.detach().flatten(),
                                   torch.tensor([-0.035, 0.035], dtype=torch.float64))
        trainer(model)
        first_class_probability = 1 / (1 + math.exp(0.07))
        expected = -0.035 - 0.07 * (0.2 + first_class_probability)
        torch.testing.assert_close(model.weight.detach().flatten(),
                                   torch.tensor([expected, -expected], dtype=torch.float64))
        self.assertEqual(trainer.epochs_completed, 2)
        self.assertEqual(trainer.last_samples, 1)
        with self.assertRaisesRegex(ValueError, 'fresh trainer'):
            trainer(copy.deepcopy(model))

    def test_empty_local_loader_is_rejected(self):
        trainer = SGDLocalTrainer([], learning_rate=0.07, momentum=0.4,
                                  weight_decay=0.0, max_grad_norm=None, device='cpu')
        with self.assertRaisesRegex(ValueError, 'no samples'):
            trainer(torch.nn.Linear(1, 2))


if __name__ == '__main__':
    unittest.main()
