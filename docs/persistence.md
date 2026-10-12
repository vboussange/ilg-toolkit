# Save a calibrated model or continue one fit

`save_model` stores the selected encoder, regional genetic calibration and
population-effect posterior, feature and target contracts, solver settings, and
observation-use provenance. The artifact supports prediction without training
measurements. `save_checkpoint` stores the latest optimization state and selected
best model for continuation; see below.

## Export predictions

Loading requires a caller-supplied encoder with the exact full configuration used
for training. Construct the same class with the same static settings, including
patch size, minimum conductance, patch batching, dropout and custom activation
choices. Its initial parameter values may differ: saved numerical leaves replace
them.

```python
import jax
from ilg_toolkit import load_model, save_model
from ilg_toolkit.models import ResNet9Conductance

save_model("model.ilg", result.model)
encoder_template = ResNet9Conductance(
    2,
    patch_size=4,
    min_conductance=0.02,
    patch_batch_size=3,
    key=jax.random.key(0),
)
model = load_model("model.ilg", encoder=encoder_template)
prediction = model.predict(prepared_query_region)
```

This example assumes the fitted encoder uses the shown settings. Matching leaf
shapes cannot establish matching static configuration; keeping that configuration
is the caller's responsibility. Compatible custom Equinox encoders use the same
save/load operations without registration or a toolkit-specific codec.

Loading does not refit calibration. Regional normalization, coefficients,
variances, posterior quantities, population identities and actual
training/validation/calibration access are retained. Marginal, known-effect and
support-conditioned predictions preserve their scientific meaning. A model may
have calibration heads for only some regions: landscape scores remain available
in compatible regions, while MLPE genetic prediction requires an explicit
regional head. See [conditioning](mlpe-conditioning.md) and
[evaluation](evaluation.md) for prediction and eligibility rules.

Query regions supply a Coordax field with named row, column and feature dimensions,
sampling-unit locations, and matching feature meanings/order. Raw landscape and
genetic measurements are not embedded in inference artifacts. Save individual
ensemble members with these same operations and assemble the desired in-memory
ensemble in an ordinary script; [examples/ensemble.py](../examples/ensemble.py)
demonstrates the ensemble workflow.

## Native format and precision

Each file contains a short JSON header followed by native Equinox leaves
of the actual `CalibratedModel` or `TrainingState`. The header holds nonnumerical
contracts and the structure needed to construct that module around the supplied encoder. Learned
coefficients and posterior arrays remain ordinary numerical leaves. There is no
architecture registry, separate numerical payload, content checksum, pickle or
exact runtime-version gate. Historical ZIP-based `.ilg` inference artifacts are
unsupported, including old training checkpoints.

Equinox checks numerical leaf compatibility with the caller's template. Invalid
headers, truncated leaves and representative structural incompatibilities raise
`ArtifactError`. Keep the original precision: artifacts with 64-bit JAX real or
integer arrays, or complex128 arrays, require `JAX_ENABLE_X64=true` or an active
`jax.enable_x64()` context when loading. Enable x64 before creating or converting
inputs when those measurements need float64 precision; enabling it later cannot
recover information already rounded. Loading never changes global JAX precision.

Saves write and flush a temporary file in the destination directory, then replace
the destination atomically. An interrupted replacement preserves the preceding
complete model. The destination's parent directory must already exist.

## Continue one fit

```python
import jax
from dataclasses import replace
from ilg_toolkit import fit, load_checkpoint, save_checkpoint

with jax.enable_x64():
    result = fit(
        region, observations, model=model, config=config,
        on_epoch=lambda state: save_checkpoint("training.ilg", state),
    )
    state = load_checkpoint("training.ilg", encoder=encoder_template)
    continued = fit(
        region, observations, state=state,
        config=replace(state.config, epochs=100),
    )
```

Supply the exact encoder configuration used for this fit, just as for `load_model`.
Include the original training/validation partitions and validation arguments when
resuming. The default budget is the saved configuration. Only the total epoch
budget may increase or stay equal; reducing the saved requested budget is rejected,
including for an intermediate checkpoint. Input shapes and precision,
sampling-unit identities/order/kinds, target meaning/units/transform, feature
meanings/order, selected training/validation pairs, and optimization/solver settings
must agree. Regional mapping order does not change this contract.

Keep the original landscape measurements, sampling locations and genetic values
unchanged yourself. Checkpoints do not fingerprint these contents: same-shape edits
cannot be detected. Do not supply a replacement model to `fit(state=...)`; the
checkpoint already contains its latest encoder. Exact deterministic continuation
is expected under the same execution conditions. Changed library versions,
devices or execution settings are not refused automatically, and may change
numerical results.

`on_epoch(state)` runs after initialization at epoch zero and after each complete
update, evaluation and model selection. Callback exceptions propagate. A fresh
zero-budget fit invokes the callback once; a continuation with no further updates
invokes it zero times. Saving at the end with `save_checkpoint(path, result.state)`
is also supported. The checkpoint retains latest encoder/nuisance parameters,
optimizer and random state, declared configuration, history and selected best
model together. Validation selection preserves the earlier best model while
continuation follows the latest optimizer trajectory. Individual ensemble members
use these same functions; `continue_members` in
[examples/ensemble.py](../examples/ensemble.py) owns filenames and loops over folds
and declared seeds without a toolkit-managed run directory.
