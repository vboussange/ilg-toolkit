# Save models and continue training

The [README quickstart](../README.md#quickstart) shows `save_model` and `load_model`.
This page covers matching model templates, numerical precision, and resumable fits.

## Model templates and precision

An `.eqx` model file retains the distance model's parameters, direct or
maximum-likelihood population effects (MLPE) calibration, feature meanings,
target specification, species order, and training and validation pairs.

Loading requires a distance model with matching class and static settings,
including patch size, patch microbatch size, dropout, and custom activation
functions. Initial parameter values may differ. Solver settings are restored
from the file, and preparation for each graph shape is rebuilt when needed.
Custom distance models use the same template pattern; see [models](models.md).

The file contains a small JavaScript Object Notation (JSON) metadata header and
Equinox-serialized array values. Equinox checks the template's array structure,
shapes, and data types. Incompatible files or templates raise `ArtifactError`.

Enable `JAX_ENABLE_X64=true` or use `jax.enable_x64()` before constructing inputs
that need float64 and before loading files containing 64-bit JAX values. Loading
preserves the current JAX precision setting. Resistance calculation and joint
MLPE fitting require float64.

The destination's parent directory must exist. Saving replaces the destination
atomically after writing and flushing a temporary file in that directory.

## Continue a fit

A checkpoint stores the latest distance parameters, optimizer and random state,
evaluation history, selected model, and selected step. With the original `region`,
`observations`, `distance_model`, and `config`:

```python
from dataclasses import replace

from ilg_toolkit import fit, load_checkpoint, save_checkpoint

result = fit(
    region, observations, distance_model=distance_model, config=config,
    on_step=lambda state: save_checkpoint("training.eqx", state),
)
state = load_checkpoint("training.eqx", distance_model=distance_model)
continued = fit(
    region, observations, state=state,
    config=replace(state.config, steps=state.config.steps + 100),
)
```

`on_step` receives state after initialization and every optimizer update. To
reproduce uninterrupted training, supply the same region batches, observations,
training and validation partitions, and validation arguments. Retain the
configuration apart from increasing its total step budget, and use the same
library versions, devices, and execution settings.

`state.model` returns the selected evaluated model. Between evaluations,
`state.latest_model` holds current distance parameters and calibration from the
preceding evaluation. Continuation uses the latest parameters with their
optimizer and random state; a fit already at its budget returns the saved state.

Save ensemble members individually with these functions. The
[ensemble example](../examples/ensemble.py) assigns checkpoint filenames by fold
and initialization seed.
