# Wishart diagnostic (experimental)

This research-only experiment scores a **fixed-parameter** Gaussian
marker-contrast model through a conductance grid graph. It is not a toolkit loss
or calibration method, and it performs no Wishart fitting.

## Assumptions

- Marker contrasts are independent and zero-mean Gaussian.
- The squared-distance matrix is complete, with explicit scatter-versus-average
  scaling and a known marker count.
- The ordinary Wishart diagnostic requires a positive-definite contrast scatter.

Under these assumptions, the diagnostic provides a reproducible reference for
comparing fixed covariance hypotheses. Missing pairs, insufficient marker rank,
or unmodeled marker dependence violate them, and structural validation cannot
establish them for real genetic observations.

## Run

From the repository root:

```bash
JAX_ENABLE_X64=true uv run --no-sync pytest experiments/wishart/test_diagnostic.py
JAX_ENABLE_X64=true uv run --no-sync python -m experiments.wishart
```

The second command demonstrates known-parameter diagnostics through a real
conductance grid graph and prints the reference and conditional held-out log
likelihoods.
