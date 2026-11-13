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
layer: a Lineax solver that a conductance model carries in `solver`. Training,
genetic prediction and calibration-head fitting use it. The
[overview](../index.md#resistance-solvers) gives JAXScape's solvers and when to
choose each.

### Solver state

JAXScape's `ResistanceDistance` prepares the solver state of every resistance
solver on a uniform conductance grid graph and reinitializes it for each
conductance surface. The solver state of JAXScape's default solver, conjugate
gradient preconditioned with geometric multigrid, holds the position of each entry
of the grounded Laplacian on the grid and the values of those entries. That of
`CuDSSSolver` holds the sparse analysis and factorization, whose analysis takes far
longer than a refactorization. That of Lineax's `CG` records only the matrix.

Training a conductance model prepares one solver state per raster shape and carries
it from update to update. The regions, species, evaluations and line searches on
one raster shape share it, so cuDSS analyzes once per raster shape. A distance
model that composes a conductance model solves through `ecological_distances`,
which prepares its solver state itself on every solve. Genetic prediction,
calibration-head fitting and `initialize_uniform_conductance` prepare a solver
state per call.

### Failures and saved models

A failed solve during training raises an error naming the region, species and
training step, and the last model saved from `on_step` remains. Constructing
`CuDSSSolver` without the `cudss` extra or a CUDA-enabled GPU backend raises
JAXScape's error.

Saved models do not record the resistance solver: the template passed to
`load_model` supplies it, so a model trained with `CuDSSSolver` on a GPU loads with
JAXScape's default solver on a CPU.
