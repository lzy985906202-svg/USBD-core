"""Integration adapter; all training and algorithm settings come from the caller."""

import copy

from usbd import flatten_params, sample_count_weights, unlearn_from_snapshots


def run_with_local_trainers(
    model,
    train_one_epoch_by_client,
    sample_counts,
    *,
    local_epochs,
    horizon,
    damping,
    singular_value_min,
    multiplier_limit,
    device,
):
    """Collect w0 and one snapshot per epoch, then perform immediate USBD.

    Each callback receives its client's model and updates it in place for one
    epoch. The caller owns the data, labels, optimizer, learning rate, batching,
    augmentation and optimizer state. Initialize optimizer state separately for
    each local model, and preserve it across that client's local epochs.
    SGDLocalTrainer in local_training.py implements this callback contract for
    ordinary classification SGD with all optimizer settings caller-supplied.
    """
    if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs <= 0:
        raise ValueError('A positive local epoch count is required')
    if not train_one_epoch_by_client or set(train_one_epoch_by_client) != set(sample_counts):
        raise ValueError('Local trainers and sample counts must identify the same clients')
    snapshots_by_client = {}
    for client, train_one_epoch in train_one_epoch_by_client.items():
        local_model = copy.deepcopy(model).to(device)
        snapshots = [flatten_params(local_model).cpu().clone()]
        for _ in range(local_epochs):
            train_one_epoch(local_model)
            snapshots.append(flatten_params(local_model).cpu().clone())
        snapshots_by_client[client] = snapshots
    return unlearn_from_snapshots(
        model, snapshots_by_client, sample_count_weights(sample_counts),
        horizon=horizon, damping=damping,
        singular_value_min=singular_value_min,
        multiplier_limit=multiplier_limit, device=device,
    )
