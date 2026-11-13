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

The resistance solver solves the linear systems of the effective-resistance layer.
A conductance model carries it in `solver`: a Lineax solver
(`lineax.AbstractLinearSolver`) that JAXScape's `ResistanceDistance(solver=...)`
accepts, with its own settings, or `None` for JAXScape's default solver,
`ResistanceDistance().solver`. A value that is neither `None` nor a Lineax solver
raises `TypeError`. The conductance model holds the solver in a static field, so
the solver holds no JAX arrays. Training, genetic prediction, calibration-head
fitting and `initialize_uniform_conductance` solve with it. JAXScape's
[solver documentation](https://vboussange.github.io/jaxscape/api/linear_solvers/)
describes its solvers, when to choose each, and what the solver state of each holds.

### Solver state

The solver state is what a resistance solver prepares for the conductance grid
graphs of one raster shape. ILG Toolkit handles it in the same way for every
resistance solver: it initializes the solver state on a uniform conductance grid
graph of the raster shape, then reinitializes it for each conductance surface
before solving.

Training a conductance model initializes one solver state per raster shape and
carries it from update to update. The regions, species, evaluations and line
searches on one raster shape share it. A solver state may hold Python objects
besides arrays; training carries its arrays and keeps its other leaves. A distance
model that composes a conductance model solves through `ecological_distances`,
which initializes a solver state on every call. Genetic prediction,
calibration-head fitting and `initialize_uniform_conductance` initialize a solver
state per call.

### Failures and saved models

A failed solve during training raises an error naming the region, species and
training step, and the last model saved from `on_step` remains.

Saved models do not record the resistance solver: the template passed to
`load_model` supplies it, so a model trained with one resistance solver on a GPU
loads with another on a CPU.
