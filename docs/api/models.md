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
The [overview](../index.md#resistance-solvers) gives the three choices and when to
choose each.

### Solver state

`"amg"` and `"cudss"` prepare a solver state on a uniform conductance grid graph:
the fixed hierarchy of `"amg"`, or the sparse analysis of `"cudss"`, which takes far
longer than a refactorization with that analysis.

Training a conductance model prepares one solver state per raster shape and carries
the state of `"cudss"` from update to update. The regions, species, evaluations and
line searches on one raster shape share it, so the analysis runs once per raster
shape. A distance model that composes a conductance model solves through
`ecological_distances`, which prepares its solver state itself and repeats the
analysis on every solve. Genetic prediction, calibration-head fitting and
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
