"""Generic classification training; no benchmark hyperparameter defaults."""

import math

import torch
from torch.nn import functional as F


class SGDLocalTrainer:
    """A one-epoch callback with optimizer state private to one client model.

    Supply a fresh instance for each client and each integration run. The caller
    supplies a loader yielding (inputs, class_indices) and all optimizer settings.
    The optimizer persists across calls, so momentum persists across local epochs.
    """

    def __init__(self, loader, *, learning_rate, momentum, weight_decay,
                 max_grad_norm, device):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Learning rate must be finite and positive')
        if not math.isfinite(momentum) or momentum < 0:
            raise ValueError('Momentum must be finite and nonnegative')
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('Weight decay must be finite and nonnegative')
        if max_grad_norm is not None and (
                not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError('Gradient norm limit must be None or finite and positive')
        self.loader = loader
        self.settings = dict(lr=learning_rate, momentum=momentum,
                             weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.optimizer = None
        self._model = None
        self.last_train_loss = None
        self.last_samples = 0
        self.epochs_completed = 0

    def __call__(self, model):
        if self._model is not None and self._model is not model:
            raise ValueError('Use a fresh trainer for a different client model')
        if self.optimizer is None:
            model.to(self.device)
            self.optimizer = torch.optim.SGD(model.parameters(), **self.settings)
            self._model = model
        model.train()
        loss_sum, samples = 0.0, 0
        for inputs, targets in self.loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite local training loss')
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            elif any(p.grad is not None and not torch.isfinite(p.grad).all()
                     for p in model.parameters()):
                raise ValueError('Non-finite local gradient')
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Non-finite local parameters')
            count = targets.shape[0]
            samples += count
            loss_sum += loss.detach().item() * count
        if samples == 0:
            raise ValueError('The local loader yielded no samples')
        self.last_samples = samples
        self.last_train_loss = loss_sum / samples
        self.epochs_completed += 1
        return self.last_train_loss
