# Wishart diagnostic (experimental)

This experiment scores a Gaussian marker-contrast model whose covariance comes
from the effective resistances of a conductance grid graph. It computes the
Wishart log likelihood and a held-out log likelihood for a fixed covariance.

## Assumptions

- Marker contrasts are independent and zero-mean Gaussian.
- The squared-distance matrix is complete, with declared average or scatter
  scaling and a known marker count.
- The contrast scatter is positive definite.

## Run

From the repository root:

```bash
JAX_ENABLE_X64=true uv run pytest experiments/wishart/test_diagnostic.py
JAX_ENABLE_X64=true uv run python -m experiments.wishart
```

The second command runs the diagnostic on a ResNet9 patch encoder with known
parameters and prints the log likelihood and the conditional held-out log
likelihood.
