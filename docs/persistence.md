# Save a calibrated model or continue one fit

`save_model` stores the selected distance model, genetic calibration module and
population-effect posterior, ordered species registry, feature and target
contracts, solver settings and observation-use provenance. Prediction needs no
original genetic measurements. `save_checkpoint` additionally stores the latest
optimization state and selected snapshot for continuation.

## Export predictions

Loading requires a caller-supplied native distance model with the exact full
configuration used for training. Learned numerical values may differ in the
template: saved leaves replace them. Architecture, ordered registry, dropout,
patch settings, conductance bounds, solver policy and supported captured custom
configuration must match. A matching array shape alone is insufficient.

```python
import jax
from ilg_toolkit import load_model, save_model
from ilg_toolkit.models import ResNet9Conductance

save_model("model.ilg", result.model)
distance_model_template = ResNet9Conductance(
    2, species=("fox",), patch_size=4, min_conductance=0.02,
    patch_batch_size=3, key=jax.random.key(0),
)
model = load_model("model.ilg", distance_model=distance_model_template)
prediction = model.predict(prepared_query_region, species="fox")["fox"]
matrix = prediction.as_matrix()
```

This assumes the fitted model uses exactly the shown settings, including the
default solver configuration. A different solver policy is rejected rather than
overwritten. Shape-specific CG/AMG preparation is rebuilt and never serialized.
Compatible custom Equinox distance models use the same operations without
registration; callers supply their native code and an explicit template.
Ambiguous opaque custom configuration fails with `ArtifactError`.

Loading does not refit calibration. Each region–species group's normalization,
coefficients, variances, posterior, population identities and actual training,
validation and calibration access are retained. Marginal, known-effect and
support-conditioned prediction preserve their scientific meaning. Compatible new
regions allow raw ecological prediction; MLPE genetic prediction requires a head
for the queried group. Direct-to-MLPE calibration replaces the entire direct
module, with no fallback for uncalibrated groups. The original
`training_objective` remains provenance. See [conditioning](mlpe-conditioning.md)
and [evaluation](evaluation.md).

Query regions supply named Coordax row/column/feature axes, species-keyed
`SamplingUnits` and matching prepared feature meanings/order. Raw landscape and
genetic measurements are not embedded in inference artifacts. Save individual
ensemble members with these operations and assemble an in-memory ensemble;
[examples/ensemble.py](../examples/ensemble.py) demonstrates this workflow.

## Native format and precision

Schema 2 contains a JSON header and native Equinox numerical leaves of the actual
`LandscapeGeneticsModel` or `TrainingState`. Header and payload integrity and
length checks detect accidental damage. Validation checks static configuration,
ordered registry, leaf shapes/dtypes and feature/target contracts. Invalid,
truncated, structurally incompatible or unsupported old artifacts raise
`ArtifactError`; there is no old-format migration. Integrity is not protection
against adversarially re-signed content and cannot prove arbitrary custom source
code equivalence. No pickle or architecture registry is needed.

Saves stream through a temporary artifact in the destination directory, flush it,
and replace the destination atomically. An interrupted replacement preserves the
preceding complete artifact. The parent directory must exist. Numerical learned
fields remain ordinary Equinox/JAX leaves.

Loading 64-bit real/integer or complex128 arrays requires x64 already enabled,
with `JAX_ENABLE_X64=true` or `jax.enable_x64()`. Loading never changes global
precision. Enable x64 before preparing resistance/MLPE inputs: enabling it later
cannot recover measurements already rounded to float32. Compatible lists, NumPy
arrays and NumPy-backed Coordax fields are accepted; JAX floating observation
dtypes are preserved, while other inputs follow configured precision.

## Continue one fit

Interrupt the original budget between evaluations to demonstrate continuation:

```python
from ilg_toolkit import TrainingConfig, fit, load_checkpoint, save_checkpoint

class Interrupted(Exception):
    pass

config = TrainingConfig(steps=4, evaluation_interval=3, seed=17)

def checkpoint_after_update(state):
    if state.step == 1:
        assert [record.step for record in state.history] == [0]
        save_checkpoint("training.ilg", state)
        raise Interrupted

try:
    fit(regions, observations, distance_model=distance_model, config=config,
        on_step=checkpoint_after_update)
except Interrupted:
    pass
state = load_checkpoint("training.ilg", distance_model=distance_model_template)
continued = fit(regions, observations, state=state)
```

Supply the original nested observations, training/validation partitions and
validation arguments when resuming. Do not supply a replacement distance model:
the checkpoint contains the latest one. The default budget is the saved
configuration. Only total `steps` may stay equal or increase; reducing the saved
requested budget is rejected even for an intermediate checkpoint. Increasing a
formerly completed budget retains its previously required final evaluation; that
is different from interruption under one fixed original budget.

Continuation verifies complete input fingerprints: landscape measurements,
sampling locations and identities/order/kinds, genetic values, target semantics,
prepared feature meanings/order, selected pairs and optimization settings must
agree. Region mapping order does not alter group identity. Keep original inputs
available; inference artifacts alone do not provide them.

`on_step(state)` observes each completed update, including updates between full
evaluations, without forcing evaluation or changing the training random stream.
Callback exceptions propagate. A fresh zero-step fit has an initial evaluation
but no callback; continuation with no updates also has no callback.
`TrainingState.step` records progress and `.history` contains sparse `StepRecord`
evaluations. Latest distance/nuisance parameters, Adam moments and RNG are separate
from the selected distance model, selected nuisance state and matching complete
calibrations. `.model` is the selected snapshot; it may precede current progress
when validation selects an earlier evaluation. Saving the returned state at the
end is also supported.

Exact deterministic continuation assumes unchanged precision and execution
conditions. There is no bitwise cross-device or cross-version guarantee.
`continue_members` in [examples/ensemble.py](../examples/ensemble.py) owns member
filenames and loops over folds/seeds without a toolkit-managed run directory.
