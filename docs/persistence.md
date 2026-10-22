# Save models and continue training

`save_model` stores a landscape genetics model in an `.eqx` file, including its
genetic calibration, population effects, feature meanings, target specification,
and training and validation pair lists. `save_checkpoint` stores a training state
for continuation, including the latest model, optimizer state, random state,
history, selected model, and selected step.

## Save and load a model

Construct a distance model with the same class and static settings used for
training, including patch size, patch microbatch size, dropout, and custom
activation functions. Its initial parameter values may differ: Equinox replaces
them with the saved values. Conductance solver settings are restored from the
file; shape-specific solver preparation is rebuilt when needed.

```python
import jax
from ilg_toolkit import load_model, save_model
from ilg_toolkit.models import ResNet9Conductance

save_model("model.eqx", result.model)
template = ResNet9Conductance(
    2,
    patch_size=4,
    min_conductance=0.02,
    patch_microbatch_size=3,
    key=jax.random.key(0),
)
model = load_model("model.eqx", distance_model=template)
prediction = model.predict(query_region)
```

The same operations support custom distance models. The caller supplies the
matching static settings; Equinox checks numerical structure, shapes, and dtypes.
Saved calibration preserves genetic predictions and population effects. Query
regions supply landscape features, sampling-unit locations, and matching feature
meanings. See [models](models.md) for model construction and
[evaluation](evaluation.md) for ensemble predictions.

## Format and precision

The first line is a JavaScript Object Notation (JSON) header containing `format`,
`kind`, `requires_x64`, and `metadata`. Metadata includes the target specification,
feature meanings, calibration information, and training and validation pair lists.
The rest of the file is written by `equinox.tree_serialise_leaves` and loaded by
`equinox.tree_deserialise_leaves` using the caller's distance model template.

Enable `JAX_ENABLE_X64=true` or use `jax.enable_x64()` before creating inputs that
need float64 precision and before loading files with 64-bit JAX values.
Resistance calculation and maximum-likelihood population effects (MLPE) fitting
require float64. Loading checks the saved precision requirement and preserves the
current JAX configuration. Incompatible files or templates raise `ArtifactError`.

Saves write and flush a temporary file in the destination directory, then replace
the destination atomically. An interrupted replacement preserves the previous
file. The destination's parent directory must exist.

## Continue a fit

```python
from dataclasses import replace
from ilg_toolkit import fit, load_checkpoint, save_checkpoint

result = fit(
    region, observations, distance_model=distance_model, config=config,
    on_step=lambda state: save_checkpoint("training.eqx", state),
)
state = load_checkpoint("training.eqx", distance_model=template)
continued = fit(
    region, observations, state=state,
    config=replace(state.config, steps=100),
)
```

Supply the original observations, region batches, training and validation
partitions, and validation arguments to reproduce uninterrupted training. Keeping
these inputs unchanged is the caller's responsibility. The configuration stays
fixed except that its total step budget may increase. State validation checks
bounds on the training step, selected step, and history length.

`on_step` receives state after initialization and each optimizer update. History
contains evaluations at step zero, every `eval_every` updates, and the final step.
`selected_model` and `selected_step` identify the retained evaluated model;
`state.model` also returns it. Between evaluations, `latest_model` contains the
latest distance parameters and the calibration from the preceding evaluation.
Continuation uses these latest parameters with their optimizer and random state.
Continuing a fit at its step budget returns the saved state.

Reproducing an uninterrupted run requires the same library versions, devices,
and execution settings. Save individual ensemble members with the same functions;
[the ensemble example](../examples/ensemble.py) demonstrates checkpoint filenames
and continuation across folds and seeds.
