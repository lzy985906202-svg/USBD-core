# USBD Core Implementation

This repository contains the multi-client USBD core for method inspection.
It is a core-only release, not a complete reproduction package for the paper's
benchmark tables. Benchmark hyperparameter values and training configurations
are not distributed in this repository.

## Included implementation

- Flattening and loading model parameters while leaving buffers unchanged.
- Exact DMD on consecutive raw local parameter snapshots, including the common
  initial model: `X = [w0, ..., w_(l-1)]`, `Y = [w1, ..., w_l]`.
- Singular-value filtering and a configurable prediction multiplier magnitude cap.
- Client influence: `E_p = Re(Phi_p Lambda_p^h pinv(Phi_p) theta) - theta`.
- Sample-count-weighted aggregation and damped subtraction:
  `theta_new = theta - damping * sum_p alpha_p E_p`.
- An adapter for caller-provided local training functions.
- A generic local SGD callback with persistent per-client momentum.
- A runnable synthetic example and analytical correctness tests.

These equations describe the implemented update; they do not by themselves
establish deletion of a client's entire training history. The influence includes
the low-rank projection residual, as in the original implementation.

## Configuration boundary

There are no benchmark defaults. The caller must explicitly supply:

| Setting | Where it is supplied |
|---|---|
| Local epoch count | `run_with_local_trainers` |
| Prediction horizon and damping | Both integration and core interfaces |
| Singular-value cutoff and multiplier cap | Both integration and core interfaces |
| Learning rate, momentum, weight decay and optional gradient clipping | `SGDLocalTrainer` or caller-owned callbacks |
| Batch size, label/data processing, shuffling and augmentation | Caller-owned data loaders |
| Target clients and their sample counts | Caller-owned mappings |
| Model architecture, input model and device | Caller-owned inputs |

Learning rate belongs to snapshot collection rather than the matrix operations
in `usbd.py`. Damping remains an explicit argument because removing damped
subtraction would change the algorithm. The supplied values affect the output;
this release does not reproduce reported benchmark results without the omitted
experiment setup. Documented settings in the manuscript remain visible to readers.

## Usage contract

Use Python 3.10 or newer and install dependencies with
`python -m pip install -r requirements.txt`. The CPU example and tests were
checked with PyTorch 2.4.1; a GPU is not required for these checks.
Import `unlearn_from_snapshots` for existing in-memory client trajectories, or
`run_with_local_trainers` for caller-managed snapshot training.

Every client trajectory starts from the same input model and uses the same
parameter order. Training callbacks update a separate local model for each
client. All required settings are keyword arguments without numerical defaults.
The core returns a copied model, client influences, DMD fits and the aggregate
influence. The input model is not modified. Model buffers are preserved; no
additional calibration or recovery training is performed by this core.

## Quick start

```bash
python -m pip install -r requirements.txt
python demo_synthetic.py
python -m unittest discover -s tests -v
```

The demo generates tiny classification datasets in memory, trains three local
models, collects parameter snapshots, and applies the USBD update. It prints
local losses, fitted ranks, the update norm and preservation checks. It does not
download data or save checkpoints. Run `python demo_synthetic.py --help` to see
the configurable example settings.

**All numerical defaults in the demo are illustrative synthetic settings, not
the paper's experimental hyperparameters.** The public library interfaces still
require explicit algorithm and optimizer settings. The demo does not establish
forgetting, modal orthogonality, or benchmark accuracy.

## Files

| File | Purpose |
|---|---|
| `usbd.py` | Exact DMD, influence prediction, weighted subtraction and parameter loading |
| `example_integration.py` | Collect the common starting snapshot and local epoch snapshots |
| `local_training.py` | Generic cross-entropy SGD callback, with separate optimizer state per client |
| `demo_synthetic.py` | Standalone synthetic example of the complete integration path |
| `tests/test_usbd.py` | Eight analytical or synthetic checks of the numerical and training contracts |
| `requirements.txt` | Runtime dependency |
| `README.md` | Algorithm, configuration and usage documentation |
| `.gitignore` | Explicit allowlist for the public release |

## Local training integration

Supply a loader for each targeted client and a fresh `SGDLocalTrainer` instance
for each run. A loader yields `(inputs, class_indices)`; the model returns class
logits. Dataset construction and any label transformation belong to the caller.
The following is an integration template: `settings`, `model`, `loaders` and
`sample_counts` must be provided by your application.

```python
from local_training import SGDLocalTrainer
from example_integration import run_with_local_trainers

trainers = {
    client: SGDLocalTrainer(
        loader,
        learning_rate=settings.learning_rate,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
        max_grad_norm=settings.max_grad_norm,  # None disables clipping
        device=settings.device,
    )
    for client, loader in loaders.items()
}
result = run_with_local_trainers(
    model, trainers, sample_counts,
    local_epochs=settings.local_epochs,
    horizon=settings.horizon,
    damping=settings.damping,
    singular_value_min=settings.singular_value_min,
    multiplier_limit=settings.multiplier_limit,
    device=settings.device,
)
updated_model = result.model
```

`sample_counts` describes each targeted client's training dataset size. Only
targeted clients enter the normalized aggregation weights. The callback reports
a sample-weighted mean training loss for its most recent epoch. Optimizer state
persists across epochs of one client; reusing a trainer with another model is
rejected. This utility supplies ordinary SGD, not the paper's complete federated
training or pollution pipeline.

## Algorithm flow

1. Copy the common model for each target client; collect `w0` and one parameter
   snapshot after each local epoch. With `l` epochs there are `l + 1` snapshots.
2. Form the raw consecutive matrices `X` and `Y`, filter singular values using
   the supplied cutoff, and fit exact DMD modes and eigenvalues.
3. Compute each influence with the stated formula and the supplied prediction
   horizon. Cap multiplier magnitudes while preserving their complex phase.
4. Normalize targeted-client sample counts and aggregate the influences.
5. Subtract the damped aggregate from a copy of the common model's parameters.
   Preserve the common model's buffers, including BN running statistics.

The original formula can have a nonzero influence at zero prediction horizon
when the low-rank modes do not fully represent the input parameters. A test
explicitly checks this projection-residual behavior. No background removal,
whitening or modal-correlation adjustment is applied.

## Validation

The tests cover a known linear dynamical system, sign-preserving multiplier
capping, the zero-horizon projection residual, sample-weighted subtraction,
input-model and BN-buffer preservation, rejection of mismatched initial
snapshots, persistent SGD momentum, and rejection of empty local loaders.
These correctness checks are separate from empirical forgetting evaluation.

## Repository scope

No comparison-method implementations, dataset downloads, model checkpoints,
snapshots, fitted modal tensors, training logs, results, parameter sweeps or
machine-specific experiment paths are included. The residual modal-correlation
diagnostic is a separate analysis and is not used in this update implementation.

## Code availability statement

> The core implementation of USBD is available at the linked repository.
> The repository provides the DMD-based client influence estimation and
> multi-client aggregation routines. Benchmark-specific training configurations
> and hyperparameter values are not included; this is not a complete benchmark
> reproduction package.
