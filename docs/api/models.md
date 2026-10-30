# Distance models

::: ilg_toolkit.models.AbstractDistanceModel

::: ilg_toolkit.models.ConductanceModel

::: ilg_toolkit.models.EmbeddingDistanceModel

::: ilg_toolkit.models.LogLinearConductance

::: ilg_toolkit.models.initialize_uniform_conductance

::: ilg_toolkit.models.ResNet9Conductance

::: ilg_toolkit.models.IBDModel

::: ilg_toolkit.models.UNetEmbeddingDistance

## Resistance solvers

The resistance solver is the numerical method behind the effective-resistance
layer. A conductance model carries its choice in `solver_config`, and training,
genetic prediction, calibration-head fitting and saved models all use that choice.

| `solver` | Method | Installation | Choose it for |
| --- | --- | --- | --- |
| `"cg"` (default) | Conjugate gradient (CG) | Included | A CPU or a GPU with no optional dependency, on small conductance grid graphs with low conductance contrast |
| `"amg"` | CG preconditioned with algebraic multigrid (AMG) | `amg` extra | A CPU or a GPU, on larger conductance grid graphs with conductance contrast below 4 orders of magnitude |
| `"cudss"` | NVIDIA CUDA Direct Sparse Solver (cuDSS) | `cudss` extra, as in [Installation](../index.md#installation) | An NVIDIA GPU, at any conductance contrast, when the factorization fits in GPU memory |

A deep patch encoder trained with the maximum-likelihood population effects (MLPE)
loss can drive conductance to its floor of 10⁻⁶. `"cg"` and `"amg"` reach their
default iteration limit on such a conductance surface, and `"cudss"` has no limit.

### Selecting a resistance solver

`ResistanceSolverConfig` selects the resistance solver. This choice uses cuDSS on a
GPU and CG elsewhere:

```python
import jax
from ilg_toolkit import ResistanceSolverConfig
from ilg_toolkit.models import ResNet9Conductance

solver_config = ResistanceSolverConfig(solver="cudss" if jax.default_backend() == "gpu" else "cg")
distance_model = ResNet9Conductance(
    in_channels=3, patch_size=8, solver_config=solver_config, key=jax.random.key(0)
)
```

`"cg"` and `"amg"` take the tolerances `rtol` and `atol` and the iteration limit
`max_steps`. `"cudss"` rejects them.

### Iterations and cost

`"cg"` and `"amg"` raise an error when they reach `max_steps`, which is 1000 by
default. The iterations they need depend on the conductance contrast:

| Conductance contrast (orders of magnitude) | `"cg"` at 100 × 100, 300 × 300 | `"amg"` at 100 × 100, 300 × 300, 1000 × 1000 |
| --- | --- | --- |
| 0 | 500, 1429 | 24, 25, 21 |
| 2 | 1522, 4962 | 145, 118, 145 |
| 4 | 6204, over 20000 | 1159, 1057, 1164 |
| 6 | over 20000, over 5000 | 9945, over 5000, over 1200 |

Grid sizes count the vertices of the conductance grid graph. Each count is the most
iterations over 20 right-hand sides at `rtol` and `atol` of 10⁻⁶, on smooth random
conductance surfaces, and "over" gives the iteration cap of that run. At the default
`max_steps`, `"cg"` reaches the limit on a uniform 300 × 300 grid and at 2 orders of
magnitude on a 100 × 100 grid, and `"amg"` reaches it at 4 orders of magnitude on
every grid size.

`"cudss"` has no iterations, and its cost does not depend on conductance contrast.
It agrees with a CPU sparse Cholesky reference to within 2.3 · 10⁻⁹ (relative) up
to 6 orders of magnitude and the floor of 10⁻⁶. Times to solve 20 right-hand sides
on an NVIDIA RTX 3090 in float64, at 4 orders of magnitude:

| Resistance solver | 100 × 100 | 300 × 300 | 1000 × 1000 |
| --- | --- | --- | --- |
| `"cg"`, with a raised `max_steps` | 1.24 s | over 23 s | – |
| `"amg"`, with a raised `max_steps` | 0.53 s | 3.0 s | 34.8 s |
| `"cudss"`, including its refactorization | 1.9 ms | 11 ms | 0.11 s |

The measurements are in
[JAXScape pull request 40](https://github.com/vboussange/jaxscape/pull/40).

### Solver state

`"amg"` and `"cudss"` prepare a solver state on a uniform conductance grid graph:
the fixed hierarchy of `"amg"`, or the sparse analysis of `"cudss"`, which takes
about 54 ms at 100 × 100, 0.37 s at 300 × 300 and 4.7 s at 1000 × 1000. Refactorizing
with that analysis takes 1.3 ms, 6.1 ms and 70 ms.

Training prepares one solver state per raster shape and carries the state of
`"cudss"` from update to update. The regions, species, evaluations and line searches
on one raster shape share it, so the analysis runs once per raster shape. Genetic
prediction, `ecological_distances`, calibration-head fitting and
`initialize_uniform_conductance` prepare a solver state per call.

### Failures and saved models

A failed solve during training raises an error naming the region, species and
training step, and the last checkpoint written by `on_step` remains. Without the
`cudss` extra or a CUDA-enabled GPU backend, preparing `"cudss"` raises an error
naming the extra.

Saved models and checkpoints record the solver settings, and loading restores them
over the template's. A model saved with `"cudss"` computes ecological distances and
genetic predictions on an NVIDIA GPU with the `cudss` extra; its conductance
surfaces need neither.

::: ilg_toolkit.ResistanceSolverConfig
