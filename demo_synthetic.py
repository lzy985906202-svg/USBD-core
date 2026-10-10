"""CPU-friendly synthetic example, NOT the paper's benchmark configuration."""

import argparse
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from example_integration import run_with_local_trainers
from local_training import SGDLocalTrainer
from usbd import flatten_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # These illustrative defaults are unrelated to paper experiment settings.
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-epochs', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--damping', type=float, default=0.25)
    parser.add_argument('--learning-rate', type=float, default=0.07)
    parser.add_argument('--momentum', type=float, default=0.4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--singular-value-min', type=float, default=1e-7)
    parser.add_argument('--multiplier-limit', type=float, default=25.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 3)).to(args.device)
    # A buffer demonstrates that the core update leaves buffers untouched.
    model.register_buffer('demo_marker', torch.tensor([17.0], device=args.device))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    teacher = torch.randn(4, 3)
    trainers, counts = {}, {}
    for client, count in enumerate((18, 24, 30)):
        generator = torch.Generator().manual_seed(args.seed + client + 1)
        inputs = torch.randn(count, 4, generator=generator)
        targets = (inputs @ teacher).argmax(dim=1)
        loader = DataLoader(TensorDataset(inputs, targets),
                            batch_size=args.batch_size, shuffle=True,
                            generator=generator)
        counts[client] = count
        trainers[client] = SGDLocalTrainer(
            loader, learning_rate=args.learning_rate, momentum=args.momentum,
            weight_decay=args.weight_decay, max_grad_norm=None, device=args.device)
    result = run_with_local_trainers(
        model, trainers, counts, local_epochs=args.local_epochs,
        horizon=args.horizon, damping=args.damping,
        singular_value_min=args.singular_value_min,
        multiplier_limit=args.multiplier_limit, device=args.device)
    report = {
        'scope': 'Synthetic integration demo; not paper results or deletion evidence',
        'synthetic_settings': vars(args),
        'client_count': len(trainers),
        'snapshots_per_client': args.local_epochs + 1,
        'parameter_count': flatten_params(model).numel(),
        'dmd_ranks': {str(c): fit.Phi.shape[1] for c, fit in result.client_dmd.items()},
        'last_local_train_losses': {str(c): t.last_train_loss for c, t in trainers.items()},
        'parameter_update_norm': (flatten_params(result.model) - flatten_params(model)).norm().item(),
        'finite_result': bool(torch.isfinite(flatten_params(result.model)).all()),
        'input_model_unchanged': all(torch.equal(v, model.state_dict()[k])
                                     for k, v in original.items()),
        'buffers_preserved': all(torch.equal(v, dict(result.model.named_buffers())[k])
                                 for k, v in model.named_buffers()),
    }
    print(json.dumps(report, indent=2))
    if not all(report[k] for k in ('finite_result', 'input_model_unchanged', 'buffers_preserved')):
        raise RuntimeError('Synthetic integration check failed')


if __name__ == '__main__':
    main()
