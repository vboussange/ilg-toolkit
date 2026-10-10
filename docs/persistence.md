# Save predictions or continue training

Choose an artifact by the operation you need after loading:

| Operation | Write / read | Preserved state | Inputs needed afterward |
| --- | --- | --- | --- |
| Predict with one fitted model | `save_model` / `load_model` | Selected encoder, regional calibration/posterior, feature and target contracts, solver settings, target-access provenance | Prepared query region; no genetic query targets or model template |
| Continue one fit | `save_checkpoint` / `load_checkpoint`, then `fit(state=...)` | Latest encoder, nuisance parameters, Adam moments/count, RNG, history, configuration and separately selected best model | Same training/validation inputs and partitions |
| Deploy an ensemble | `save_ensemble` / `load_ensemble` | Every requested identity, fold and outcome, with each completed `member.model` | Prepared query region; no checkpoints or training observations |
| Continue an ensemble run | `fit_ensemble_run(..., resume=True)` | Atomic manifest referring to independent member checkpoints and selected models | Complete run directory and same accessed training/validation inputs |

All operations use the public `ilg_toolkit` boundary. The consolidated
[`examples/ensemble.py`](../examples/ensemble.py) demonstrates independently fitted
members, deployment, known effects, declared support and eligible OOF evaluation.

## Export predictions

```python
from ilg_toolkit import load_model, save_model, load_ensemble, save_ensemble

save_model("model.ilg", result.model)
model = load_model("model.ilg")
prediction = model.predict(prepared_query_region)

save_ensemble("ensemble.ilg", ensemble)
restored = load_ensemble("ensemble.ilg")
prediction = restored.predict(prepared_query_region)
```

Loading does not refit calibration. Regional posterior effects, target transforms,
actual training/validation/calibration access and constructor settings travel with
the encoder. Query regions supply HWC features, native row/column grid locations,
and matching feature meanings/order; preparation and normalization remain the
caller's responsibility. Raw landscape scores work in compatible transfer regions;
MLPE target prediction requires an explicit regional head. Explicit recalibration
of `member.model` is retained when exporting an ensemble. Inference reloads have
`member.fit_result is None` and contain no optimizer continuation state.

Ensemble deployment averages each member's predictions **after** genetic
calibration and target inversion. It requires all requested members; failed and
pending outcomes remain explicit. Member SD is descriptive, in original target
units. Conditional MLPE predictive variance remains separate on the transformed
model scale. See [evaluation](evaluation.md) for eligible OOF aggregation and
[conditioning](mlpe-conditioning.md) for known-effect and supplied-support access.

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
    state = load_checkpoint("training.ilg")
    continued = fit(
        region, observations, state=state,
        config=replace(state.config, epochs=100),
    )
```

Include the original training/validation partitions and validation arguments when
resuming. The default budget is the saved configuration. Only the total epoch
budget may increase or stay equal; reducing the saved requested budget is rejected,
including for an intermediate checkpoint. Selected measurements, landscapes,
sampling-unit identities/locations/kinds, target and feature contracts, partitions,
model overrides and other optimization/solver settings must agree. Equivalent
regional mapping order does not change identity. Unaccessed query measurements do
not determine continuation identity.

`on_epoch(state)` runs after initialization at epoch zero and after each complete
update, evaluation and model selection. Callback exceptions propagate. A fresh
zero-budget fit invokes the callback once; a continuation with no further updates
invokes it zero times. Saving at the end with `save_checkpoint(path, result.state)`
is also supported. The checkpoint retains latest and best state together, so
validation selection does not replace the trajectory used for continuation.

## Continue independent ensemble members

```python
from ilg_toolkit import TrainingConfig, fit_ensemble_run

with jax.enable_x64():
    ensemble = fit_ensemble_run(
        "training-run", region, observations,
        n_folds=2, holdout_size=2, fold_seed=47,
        initialization_seeds=(13, 29),
        config=TrainingConfig(objective="mlpe", epochs=30),
    )
    ensemble = fit_ensemble_run("training-run", region, observations, resume=True)
```

Use one writer per run directory. A fresh call refuses an existing manifest.
Resume validates the complete saved composition and referenced artifacts before
training. It skips compatible members already at budget and continues unfinished
members from their own latest state. An explicit larger total budget continues
previously completed members. Fold definitions, initialization identities,
accessed data and configuration must agree. Factories initialize pending members
only; their model family, constructor settings, leaf shapes/dtypes and dynamic
architecture settings must match the run.

Each complete epoch writes an immutable member generation before atomic
replacement of `run.ilg` publishes its reference. Completed outcomes also reference
the selected inference model, verified against the checkpoint's best model.
`on_epoch(identity, state)` runs after durable epoch publication;
`on_member(member)` runs after durable outcome publication, including skipped
members. Callback errors interrupt execution. Initialization and fit failures are
recorded with diagnostics, and later requested members still run. Resume retains
those failures; start a separate run for a changed retry request.

Missing, corrupt or incompatible references raise `ArtifactError` naming the
member before work resumes. A failed publication preserves the preceding complete
manifest and its generations. Unreferenced files may remain after interruption;
resume never guesses a replacement. Keep every referenced generation for
continuation. References are restricted to the run's `members` directory and
checked against traversal and symlink escapes.

## Format, compatibility and precision

The four artifact kinds use schema-one ZIP archives with strict finite JSON
metadata and numeric NPY arrays; no pickle or arbitrary class imports. Loaders
validate kinds, schemas, codec versions, leaf paths/count/shapes/dtypes, declared
array digests and complete composition. Missing, extra, duplicate or malformed
content raises `ArtifactError`. Integrity checks detect damage, not authorship.
Saves write a same-directory temporary archive, flush it, then atomically replace
the destination. A failed save preserves the preceding complete generation.
Single-file destinations require an existing parent directory; saved runs create
their directories.

Codecs support the exact shipped `UNetEmbeddingDistance` and `ResNet9Conductance`
classes with stateless GroupNorm. Unsupported custom architectures, architecture
modifications or mutable model state require explicit future codecs. Inference
artifacts may contain heads for only selected regions; checkpoints require heads
for every MLPE training region.

ResNet9 retains its optional `patch_batch_size` execution setting in models,
checkpoints and saved runs. Default unbatched models keep the historical ResNet
codec version 1 metadata; explicitly configured positive batch sizes use version
2. Both load strictly under archive schema 1. A version 1 ResNet loads with
`patch_batch_size=None`. Continuation uses the saved encoder and its batch size;
it rejects a replacement model supplied alongside the continuation state.

Array dtypes are preserved. Float64 JAX leaves require `JAX_ENABLE_X64=true` or an
active `jax.enable_x64()` context when loading; silent truncation is refused.
MLPE fitting and exact checkpoint continuation require the saved explicit
precision setting. Exact resume also requires the recorded Python/numerical
library versions, backend and device kind/count under deterministic execution;
incompatible runtimes are refused. Solver contexts are rebuilt from configuration.
Inference reconstruction verifies architecture and numerical contracts without
promising identical outputs across future runtimes or devices. Matching
deterministic CPU round trips check numeric-array identity and prediction
agreement; installed-wheel comparisons also check floating-point tolerances.
