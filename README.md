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
| Learning rate, momentum, weight decay, batch size and label/data processing | Caller-owned local training callbacks |
| Target clients and their sample counts | Caller-owned mappings |
| Model architecture, input model and device | Caller-owned inputs |

Learning rate belongs to snapshot collection rather than the matrix operations
in `usbd.py`. Damping remains an explicit argument because removing damped
subtraction would change the algorithm. The supplied values affect the output;
this release does not reproduce reported benchmark results without the omitted
experiment setup. Documented settings in the manuscript remain visible to readers.

## Usage contract

Install dependencies with `python -m pip install -r requirements.txt`.
Import `unlearn_from_snapshots` for existing in-memory client trajectories, or
`run_with_local_trainers` for caller-managed snapshot training. These are library
interfaces, not a preconfigured benchmark command.

Every client trajectory starts from the same input model and uses the same
parameter order. Training callbacks update a separate local model for each
client. All required settings are keyword arguments without numerical defaults.
The core returns a copied model, client influences, DMD fits and the aggregate
influence. The input model is not modified. Model buffers are preserved; no
additional calibration or recovery training is performed by this core.

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
