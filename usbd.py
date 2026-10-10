"""USBD multi-client core with caller-supplied configuration.

This module contains no benchmark configurations or trained model weights.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Hashable

import torch


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    """Flatten parameters only; model buffers are excluded."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


@torch.no_grad()
def load_params_to_model(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Load a parameter vector without recalibrating or changing buffers."""
    if flat.ndim != 1 or flat.numel() != sum(p.numel() for p in model.parameters()):
        raise ValueError('Parameter vector does not match the model')
    if not torch.isfinite(flat).all():
        raise ValueError('Non-finite parameter vector')
    offset = 0
    for param in model.parameters():
        count = param.numel()
        param.copy_(flat[offset:offset + count].reshape(param.shape))
        offset += count


class DMDUnlearner:
    """Exact DMD fitted to raw consecutive parameter snapshots, including w0."""

    def __init__(
        self,
        snapshots: Sequence[torch.Tensor],
        device: str | torch.device,
        *,
        singular_value_min: float,
        multiplier_limit: float,
    ):
        if len(snapshots) < 2:
            raise ValueError('At least two snapshots are required')
        if not math.isfinite(singular_value_min) or singular_value_min < 0:
            raise ValueError('Invalid singular-value cutoff')
        if not math.isfinite(multiplier_limit) or multiplier_limit <= 0:
            raise ValueError('Invalid prediction multiplier limit')
        first = snapshots[0]
        if any(s.ndim != 1 or s.shape != first.shape or s.dtype != first.dtype
               or not s.is_floating_point() or not torch.isfinite(s).all()
               for s in snapshots):
            raise ValueError('Snapshots must be finite, aligned real parameter vectors')
        self.device = device
        self.multiplier_limit = multiplier_limit
        self.X = torch.stack(list(snapshots[:-1]), dim=1).to(device)
        self.Y = torch.stack(list(snapshots[1:]), dim=1).to(device)
        self.Phi, self.Lambda = self._compute_dmd(singular_value_min)

    def _compute_dmd(self, singular_value_min):
        U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
        valid = S > singular_value_min
        if not valid.any():
            raise ValueError('No singular value survives the supplied cutoff')
        U, S, Vh = U[:, valid], S[valid], Vh[valid, :]
        S_inv = torch.diag(1.0 / S)
        V = Vh.mH
        Atilde = U.mH @ self.Y @ V @ S_inv
        eigenvalues, eigenvectors = torch.linalg.eig(Atilde)
        lifted = self.Y @ V @ S_inv
        Phi = lifted.to(eigenvectors.dtype) @ eigenvectors
        return Phi, eigenvalues

    def get_influence_vector(self, current_weights_flat, h_horizon):
        """Return Re(Phi Lambda^h pinv(Phi) theta) - theta.

        The supplied magnitude cap preserves the complex multiplier phase.
        Float32 output follows the experiment core's parameter update path.
        """
        if isinstance(h_horizon, bool) or not isinstance(h_horizon, int) or h_horizon < 0:
            raise ValueError('Prediction horizon must be a nonnegative integer')
        if current_weights_flat.ndim != 1 or current_weights_flat.numel() != self.X.shape[0]:
            raise ValueError('Current parameters do not match the snapshots')
        theta = current_weights_flat.to(self.device).to(self.Phi.dtype)
        multipliers = torch.pow(self.Lambda, h_horizon)
        magnitude = multipliers.abs()
        overflow = magnitude > self.multiplier_limit
        if overflow.any():
            multipliers[overflow] *= self.multiplier_limit / magnitude[overflow]
        coefficients = torch.linalg.pinv(self.Phi) @ theta
        predicted = self.Phi @ (torch.diag(multipliers) @ coefficients)
        influence = predicted.real.float() - theta.real.float()
        if not torch.isfinite(influence).all():
            raise ValueError('Non-finite DMD influence')
        return influence


def sample_count_weights(sample_counts: Mapping[Hashable, int]) -> dict:
    """Normalize targeted clients' sample counts to sum to one."""
    if not sample_counts or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                                for n in sample_counts.values()):
        raise ValueError('Positive integer client sample counts are required')
    total = sum(sample_counts.values())
    return {client: count / total for client, count in sample_counts.items()}


def aggregate_and_subtract(theta, influences, weights, *, damping):
    """Compute theta - damping * sum_p weight[p] * influence[p]."""
    if not influences or set(influences) != set(weights):
        raise ValueError('Client influences and aggregation weights must match')
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Invalid aggregation weights')
    tolerance = len(weights) * torch.finfo(torch.float64).eps
    if abs(math.fsum(weights.values()) - 1.0) > tolerance:
        raise ValueError('Aggregation weights must sum to one')
    if not math.isfinite(damping) or damping < 0:
        raise ValueError('Invalid damping')
    delta = torch.zeros_like(theta)
    for client in influences:
        influence = influences[client]
        if influence.shape != theta.shape:
            raise ValueError('Influence vector shape mismatch')
        delta += weights[client] * influence.to(theta.device)
    result = theta - damping * delta
    if not torch.isfinite(delta).all() or not torch.isfinite(result).all():
        raise ValueError('Non-finite aggregated update')
    return result, delta


@dataclass
class USBDResult:
    model: torch.nn.Module
    aggregated_influence: torch.Tensor
    client_influences: dict
    client_dmd: dict


def unlearn_from_snapshots(
    model,
    client_snapshots,
    client_weights,
    *,
    horizon,
    damping,
    singular_value_min,
    multiplier_limit,
    device,
):
    """Apply multi-client USBD to a copy of a common starting model.

    Each client's first snapshot must equal that common model's parameters.
    Raw snapshots are used, without centering, whitening or mode pruning.
    No recovery training or additional buffer calibration is performed.
    """
    if not client_snapshots or set(client_snapshots) != set(client_weights):
        raise ValueError('Snapshots and weights must identify the same clients')
    result_model = copy.deepcopy(model).to(device)
    theta = flatten_params(result_model)
    influences, fits = {}, {}
    for client, snapshots in client_snapshots.items():
        if not snapshots or not torch.equal(snapshots[0].to(device), theta):
            raise ValueError('Every client must start from the same global model')
        dmd = DMDUnlearner(snapshots, device,
                           singular_value_min=singular_value_min,
                           multiplier_limit=multiplier_limit)
        influences[client] = dmd.get_influence_vector(theta, horizon)
        fits[client] = dmd
    updated, delta = aggregate_and_subtract(theta, influences, client_weights,
                                            damping=damping)
    load_params_to_model(result_model, updated)
    return USBDResult(result_model, delta, influences, fits)
