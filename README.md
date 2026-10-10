# ILG Toolkit

An independent Python toolkit for inverse landscape genetics. Fit landscape
encoders from prepared arrays and labelled pairwise genetic observations, then
predict at query sampling-unit locations without their genetic observations.

## Install

Python 3.12 or newer is required. From this checkout:

```bash
pip install .
# Optional geospatial preparation libraries: pip install '.[geo]'
# Optional AMG preconditioning: pip install '.[amg]'
# Optional CUDA JAX wheels: pip install '.[gpu]'
# Tests and formatting: pip install '.[dev]'
```

The minimal workflow uses NumPy, SciPy, JAX, Equinox, Optax, Lineax, and JAXScape.
It has no paper-data,
benchmark, geospatial acquisition, or research-checkout dependency.

## Synthetic quickstart

Run `JAX_ENABLE_X64=true python examples/quickstart.py` after installing. The complete in-memory
workflow is:

```python
import numpy as np
from ilg_toolkit import TrainingConfig, PairwiseObservations, RegionBatch, TargetSpec, fit

region = RegionBatch(
    name="my-catchment",
    features=np.arange(32, dtype=np.float32).reshape(4, 4, 2) / 32,
    sampling_unit_ids=("a", "b", "c", "d"),
    grid_positions=np.array([[0, 0], [0, 3], [3, 0], [3, 3]]),
)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, .2, .4, .6], [.2, 0, .2, .4], [.4, .2, 0, .2], [.6, .4, .2, 0]],
    target=TargetSpec("genetic dissimilarity", units="index"),
)
result = fit(region, observations, config=TrainingConfig(epochs=30, learning_rate=.01))
prediction = result.model.predict(region)
print(prediction.values, prediction.target)
scores = result.model.landscape_scores(region)
```

`RegionBatch` keeps prepared landscape inputs separate from observations.
`TrainingConfig` declares optimization settings, and `result.model` is a
`CalibratedModel` retaining the encoder, target contract, resistance settings,
and any regional MLPE heads. Direct regression uses the declared target mapping
without an MLPE head; label-free landscape scoring is available in either mode.

`predict()` returns a symmetric matrix in the region's sampling-unit order. For
selected pairs, use `result.model.predict_pairs(region, [("a", "d")])`.
Its `PairPrediction` contains a vector of `values`, the requested `pairs` in their
original order/orientation, `target`, `region_name`, and `scale="original"`.
Pairs must be nonempty, distinct and unique as unordered identities. Both direct
and MLPE models select scores before target inversion; unrequested pairs do
not trigger inverse-transform failures. MLPE pair predictions are marginal.

Features have shape `(rows, columns, channels)` and must already be prepared;
locations are explicit integer `(row, column)` positions. Labels declare the
matrix row/column identity and are aligned to the region before training.
Fitting performs no coordinate conversion, normalization, FST conversion, or
observation clipping. Query regions contain features and locations only.

`RegionBatch.features` is a JAX-backed Coordax field with named `row`, `column`,
and `feature` axes; `feature_array` exposes its canonical HWC device array.
Pass a labelled Coordax field in any named-axis order to retain its declared
coordinates. Feature labels must match `feature_names` in meaning and order;
conflicting declarations raise. Plain prepared arrays remain accepted.
Numerical model outputs and target transformations are JAX arrays. Original
observations retain float64 precision on the host; fitting or transforming
float64 targets requires explicit `JAX_ENABLE_X64=true`. Input construction
does not change the global JAX precision setting.

The initial model is the cleaned architecture's U-Net embedding-distance family.
Its landscape scores are squared Euclidean distances between learned patch
embeddings. Direct regression trains these distances with nonnegative
`log1p`-MSE and returns them on the declared original target scale: `log1p` is a
loss transform, not a target transform. Signed relatedness is unsupported by
this objective. Direct regression requires no MLPE calibration head.

Pass a configured `ilg_toolkit.models.UNetEmbeddingDistance` as `model=` to
choose patch size, embedding dimensions, U-Net width, and dropout explicitly.
For example, `patch_size=1, base_channels=2, embedding_dim=2, dropout=0` is useful
for tiny synthetic problems. Raster dimensions must be divisible by patch size.

Custom Equinox encoders implement the abstract `ConductanceModel.conductance`
or `EmbeddingDistanceModel.embedding_grid` boundary and declare `patch_size`.
Mark concrete implementations with `typing.final`; adapt a concrete encoder by
composition. The shipped ResNet9 and U-Net are final implementations.

Without validation, the final fixed-budget encoder is returned. With
`validation=(prepared_validation_region, validation_observations)`, the lowest
validation log1p-MSE selects initialization or an updated encoder. Validation
targets never enter optimization, and test data is not required. The result
records every epoch's inference-mode training/validation objective and the
selected epoch. The optimizer uses fixed-rate Adam; no validation-dependent
schedule or early stopping is implicit.

## Conductance and resistance

Run `JAX_ENABLE_X64=true python examples/conductance.py` for a synthetic ResNet9
fit through the real graph solver. Pass a configured
`ilg_toolkit.models.ResNet9Conductance(in_channels=..., patch_size=4, key=...)`
to `fit`. The raster must be divisible into patches, and its patch grid must
contain at least two vertices. Sampling units within one patch have zero
resistance to each other. Feature counts are supplied explicitly.

The encoder uses the cleaned ResNet9 architecture with stateless GroupNorm and
an explicit `min_conductance` floor (default `1e-6`) after a stable softplus head.
It imposes no upper conductance bound. `model.conductance_surface(region)`
returns the fitted patch surface; `landscape_scores(region)` returns effective
resistance on a four-neighbour graph with mean endpoint conductances.
`predict(region)` returns direct-regression predictions on the declared target
scale. These are distinct outputs; accurate target predictions alone do not
establish that a unique biological conductance surface has been recovered.

Graph solves require JAX float64 to be enabled explicitly before Python starts;
CNN parameters and outputs remain float32. Configure `TrainingConfig(solver=ResistanceSolverConfig(
rtol=1e-6, atol=1e-6, max_steps=1000, use_amg=False))` to choose convergence
settings. AMG is optional and builds a reusable hierarchy outside differentiation.
Failure to converge raises with the region and epoch; the toolkit does not
silently change solver settings. Prediction retains the fitted solver configuration.

## Development

```bash
pytest
ruff check .
```

Tests exercise public fit/predict behavior on synthetic arrays and build a wheel
to run outside the source tree. Extracted components and their license are listed
in `NOTICE`; study-specific code remains in its original repositories.

## Labelled pairs, transformations, and partitions

Incomplete observations need no invented genetic targets:

```python
from ilg_toolkit import ObservationPartition

observations = PairwiseObservations.from_pairs(
    [("a", "b"), ("a", "d")], [.2, .6],
    target=TargetSpec("genetic dissimilarity", units="index", transform="log1p"),
)
training = ObservationPartition(region.name, (("a", "b"),), role="training")
validation = ObservationPartition(region.name, (("a", "d"),), role="validation")
result = fit(
    region, observations, partition=training,
    validation=(region, observations), validation_partition=validation,
    config=TrainingConfig(epochs=10),
)
```

Explicit pair inputs require finite values and distinct, known endpoints; reversed
pairs represent the same observation and cannot be duplicated. A matrix may use
symmetric off-diagonal `NaN` entries for absent pairs. `observed_pairs` and
`observed_values` expose measured observations only. Their original measurement
precision is retained. Partition selections must be nonempty, refer to observed
pairs, and match their region and requested role. Training and validation pairs
in the same declared region must be disjoint, with or without partition arguments.

The default target transform is `identity`. Explicit `log1p` and `sqrt` transforms
require nonnegative observations. They map measurements into the encoder's
regression scale; `predict()` applies their inverse and reports `scale="original"`
with the original `TargetSpec.units`. `landscape_scores()` exposes encoder scores
before this inverse operation. A square-root inverse rejects negative transformed
predictions rather than silently squaring them. No FST linearization or target
clipping is automatic. Target kinds distinguish dissimilarity, similarity, and
relatedness; direct regression accepts nonnegative dissimilarity only.

Declare `RegionBatch(feature_names=("elevation", "canopy"), ...)` to record
feature meanings and channel order. Prediction and validation require that same
contract; equal channel counts alone do not validate declared feature meanings.
An unnamed feature contract remains available for the simplest single-region
prepared-array workflow. `sampling_unit_kinds` optionally supplies one
`"population"` or `"individual"` entry per label (default: population). This
identity metadata reserves individual-based use without implementing or asserting
statistical validity for individual-relatedness models.

## Shared encoders across regions

`fit` accepts aligned sequences or mappings keyed by the exact declared region
names for both regions and observations. The simplest single-region call stays
the same. Run `python examples/shared_regions.py` for a synthetic U-Net example
with different regional raster extents.

```python
result = fit(
    {"headwaters": upstream_region, "lowlands": downstream_region},
    {"headwaters": upstream_observations, "lowlands": downstream_observations},
    config=TrainingConfig(epochs=30),
)
upstream_prediction = result.model.predict(upstream_region)
downstream_prediction = result.model.predict(downstream_region)
```

Different training or validation regions must declare identical `feature_names`,
preserving channel meanings and order, and compatible target metadata. Equal channel counts alone
are insufficient. Mapping keys must match region names exactly; duplicate names
are rejected. Optional `partition` and `validation_partition` take aligned
sequences or mappings in the same form; use `None` entries to select all observed
pairs for a particular region. Validation takes `(regions, observations)` and may
cover fewer regions. It must preserve the training feature and target contracts.

Each regional contribution is its mean over selected observed pairs. The shared
objective is the equal-weight mean of those regional contributions. A region
with more observed pairs therefore does not automatically receive more weight.
Training finishes each regional backward pass before starting the next, retaining
only the accumulated parameter gradients, then applies one shared Adam update.
Model encoders remain stateless apart from their learned Equinox parameters.

Region names are sorted before assigning training random keys. Reordering input
mappings or aligned sequences leaves the stochastic training trajectory unchanged.
Validation uses inference mode, consumes no training random keys, and selects the
encoder without altering optimization. `result.region_names` records training
region identities; each epoch exposes `training_by_region` and
`validation_by_region` alongside the equal-region aggregate objectives.


## Regional MLPE calibration

`python examples/mlpe_calibration.py` demonstrates a frozen encoder's landscape
scores and a separate full-Gaussian-ML genetic calibration:

```python
from ilg_toolkit import calibrate_mlpe

# One finite score per observations.observed_pairs, in that documented order.
head = calibrate_mlpe(scores, observations, region_name="my-catchment")
predictions = head.predict_marginal(query_scores, query_pairs)
```

The head profiles a signed intercept and slope by GLS, standardizes training
scores with sample SD (`ddof=1`), and fits positive population-effect and residual
variance components. `MLPEConfig` makes variance floor, score-scale threshold,
explicit jitter, and optimizer budget configurable. Calibration uses float64;
JAX likelihood kernels follow input precision without changing global settings.
Constant scores, unidentified variance components, and failed optimization are
reported explicitly. Relatedness and similarity require a dedicated observation
model and are rejected by this population MLPE boundary.

Scores align with the observations' deterministic `observed_pairs` order, even
when pair input was supplied in another order. An optional `ObservationPartition`
selects declared training, validation, or calibration observations; their pair
identities and roles are retained on the head. Missing pairs are absent from the
likelihood. Query and support partitions cannot calibrate the head.

Marginal prediction uses zero-mean population effects, including for unseen
populations. It accepts labelled query pairs and scores, without query targets.
`predictions.values` reports original target units; `model_values` retains the
fitted transformed scale. With a nonlinear target transform, inverse-transforming
the fitted mean is not a distributional mean correction. Negative signed
predictions are retained without clipping. Explicit jitter is consistently added
to residual variance in both likelihood and the stored effect posterior.

## Frozen encoder recalibration and regional transfer

Run `python examples/region_transfer.py` for a synthetic U-Net transfer example.
Recalibration is an explicit operation returning a new model:

```python
from ilg_toolkit import recalibrate

# Default selection uses only the encoder's recorded training pairs.
calibrated = recalibrate(result.model, region, observations)

# Deliberately add validation measurements as development data.
developed = calibrated.recalibrate(
    region, observations, partitions=(training_partition, validation_partition),
)
```

A direct model continues to predict without any MLPE head. Explicit
recalibration converts the returned model to MLPE mode, freezing the exact
encoder object. Each MLPE region then requires its own head. `landscape_scores`
works in an unseen region with only prepared features and locations;
`predict` there raises until explicit regional calibration is supplied:

```python
development = ObservationPartition(
    new_region.name, new_observations.observed_pairs, role="calibration",
)
transferred = calibrated.recalibrate(
    new_region, new_observations, partitions=development,
)
new_predictions = transferred.predict(new_region)
```

New-region transfer requires declared matching feature meanings and ordering and
compatible target meaning, units, and transform. Population MLPE calibration and
prediction reject individual-kind sampling units; generic landscape scoring is
still available. MLPE prediction is marginal by default, including for unseen
populations in a calibrated region. It returns original-scale marginal predictions without clipping, with a
structural zero diagonal in the matrix interface. Nonlinear inverse transforms
retain the fitted-mean interpretation described above.

Explicit selections may declare training, validation, or calibration roles.
Query and support roles cannot recalibrate a head; overlapping selections are
rejected. Every head records selected pair identities and per-pair data roles.
`CalibratedModel.training_pairs` and `validation_pairs` independently retain encoder
training and selection access, including after subsequent recalibration. A new
region with no recorded encoder-training pairs requires an explicit selection.
Existing head numerical settings are retained unless a replacement `MLPEConfig`
is supplied. The original model, other regional heads, and prior fit history
remain unchanged; new development-data use does not rewrite earlier validation.

## Joint MLPE training

Select `TrainingConfig(objective="mlpe")` to train an embedding or conductance encoder
with full Gaussian maximum likelihood. Enable JAX float64 explicitly:

```python
import jax
from ilg_toolkit import TrainingConfig, fit

with jax.enable_x64():
    result = fit(
        region, observations, model=model,
        config=TrainingConfig(objective="mlpe", epochs=30, learning_rate=.01),
    )
    genetic_prediction = result.model.predict(region)
    landscape_scores = result.model.landscape_scores(region)
```

The same call accepts regional sequences or mappings, with one shared encoder
and separate population and residual variances per region. Each Adam update
averages per-pair regional likelihood gradients. Intercept and slope are signed
GLS profiles; both variance parameters use the same learning rate as the encoder.
There is no converged inner variance fit at each update. Initial variances can be
declared as `mlpe_initial_variances=(population_variance, residual_variance)`;
otherwise each region starts from its training target's sample variance (one
quarter and one half, respectively). `mlpe_variance_floor` and `mlpe_jitter` are
explicit, with no automatic jitter increase.

After every update, training observations refresh the score mean, sample SD,
GLS coefficients and population-effect posterior. Validation uses those frozen
training quantities and can contain one observed pair. It requires a training
calibration for the same region and cannot overlap training pairs. Validation
selects the returned model without changing the fixed epoch budget or
learning rate. A trained head reports `converged=False`: the budget does not
claim a fully converged variance optimum. Constant scores, unidentifiable pair
sets, nonfinite gradients or posteriors, and unsupported variance conditioning
stop fitting with the region and epoch in the diagnostic. MLPE training currently
supports population dissimilarities.

`result.state` retains the latest encoder, regional raw variances, Adam state,
random key, history, selection and fixed learning-rate/stopping policy.
`result.latest_model` can differ from the validation-selected
`result.model`. In-memory continuation preserves that trajectory:

```python
from dataclasses import replace

with jax.enable_x64():
    continued = fit(
        region, observations, state=result.state,
        config=replace(result.state.config, epochs=60),
    )
```

Continuation requires identical selected observations, landscapes, target and
feature declarations, validation inputs and configuration; only the total epoch
budget may increase. Include the same partitions and validation arguments when
resuming. The state uses a legacy uint32 JAX random key; solver contexts are
rebuilt from the declared solver settings. [Training checkpoints](docs/checkpoints.md)
describe atomic disk save/load, per-epoch checkpoint callbacks, and compatibility.

Save a complete fitted model for inference in another process:

```python
from ilg_toolkit import load_model, save_model

save_model("fitted.ilg", result.model)
model = load_model("fitted.ilg")
prediction = model.predict(prepared_query_region)
```

The artifact retains feature and target contracts, solver settings, regional
calibrations, population-effect posteriors and observation-use provenance.
It supports the shipped U-Net and ResNet9 encoders and is separate from a
resumable training checkpoint. See [the inference artifact contract](docs/inference-artifacts.md).

## Independent fold ensembles

Run `python examples/ensemble.py` for actual independent U-Net/MLPE fits and
`python examples/ensemble_averages.py` for known-output transform and graph checks.

```python
from ilg_toolkit import fit_ensemble

with jax.enable_x64():
    ensemble = fit_ensemble(
        region, observations, n_folds=2, holdout_size=2, fold_seed=47,
        initialization_seeds=(13, 29), model_factory=model_factory,
        config=TrainingConfig(objective="mlpe", epochs=30),
    )
    prediction = ensemble.predict(region)
```

`model_factory(key)` constructs a fresh encoder for each member; omitting it uses
the default U-Net. Every fold/seed member has its own initialized parameters,
Adam state, calibration and history, and members train sequentially. Member
identities and effective uint32 initialization seeds are stable across input
ordering. The effective seed derives from SHA256 of the fold ID and declared
initialization seed; `ensemble_member_identity` exposes this mapping. If seeds
are omitted, the one declared initialization seed is `config.seed`.

`generate_population_folds` draws reproducible repeated population holdouts
independently per region; folds can overlap. `holdout_size` accepts an integer or
a mapping by region. Training uses measured pairs with neither endpoint held
out. Default query pairs have both endpoints held out; declare
`query_regime="at_least_one_unseen"` for pairs with at least one held-out endpoint.
No held-out targets select a model by default. Empty measured training/query
partitions fail clearly, rather than filling missing observations or retrying a
different split. MLPE still requires at least three informative training pairs.

For a supplied design, pass `folds=[PopulationFold(...), ...]` with named regional
`held_out_units`, `training`, `query` and optional `validation` partitions.
Validation can explicitly reuse query pairs for model selection; the model
records that target access. A nominal holdout assignment then does not establish
out-of-fold eligibility. `ensemble.predict` is deployment prediction and uses
all members; eligibility-aware evaluation is a separate operation.

The aggregate averages each member's original-target-scale marginal prediction
**after** its own genetic calibration and inverse transformation.
`prediction.member_values`, `.member_ids`, and `.member_spread` retain individual
outputs and descriptive population SD (`ddof=0`). This spread is not a confidence
interval or MLPE predictive variance. Targets, units, transforms, sampling-unit
ordering and feature contracts must agree. `ensemble.landscape_scores(region)`
returns raw scores separately by member. `ensemble.conductance_surfaces(region)`
returns descriptive mean/spread for conductance families; genetic predictions do
not solve resistance on that mean surface. Conductance 1 and 4 give resistances
1 and .25 on a single edge: mean resistance .625 differs from resistance .4 on
the mean surface 2.5.

`ensemble.members` includes every requested outcome and its identity/fold,
`model`, `fit_result` or explicit `failure`. `.failures` reports failed members;
prediction refuses to drop failed, pending or missing members. Initialization or
numerical fit failures are recorded and remaining requested members still run.
`on_member(member)` observes each sequential outcome. Its exceptions propagate
so an interrupted save does not masquerade as a successful reduced ensemble.

For external durable orchestration, `fit_ensemble_member` fits or continues one
member using `state=TrainingState`. `fit_ensemble(member_states={member_id: state})`
continues specified members in memory. `on_epoch(identity, state)` receives each
complete member epoch and can save that member's checkpoint; callback failures
interrupt execution. Solver contexts and differentiation graphs are never
retained across independent member fits.

`save_ensemble(path, ensemble)` and `load_ensemble(path)` preserve every member
and its deployment model in one portable inference artifact.
`fit_ensemble_run(directory, region, observations, ...)` saves independent
member checkpoints and an atomic run manifest. Resume with the same inputs and
`resume=True` to skip compatible completed members and continue unfinished
ones. An explicit larger total epoch budget continues completed members from
their saved state. Missing, corrupt, incompatible or failed members remain
explicit. See [ensemble persistence](docs/ensemble-persistence.md) for usage,
compatibility, callbacks and artifact composition.

## Out-of-fold evaluation

Use `predict_out_of_fold` for label-free eligible predictions,
`score_out_of_fold` to read query targets afterward, or `evaluate_ensemble` for
both steps. Eligibility follows nominal endpoint holdouts and recorded encoder
training, validation selection, calibration and declared support access. Eligible
members are averaged before each unique region/pair contributes once to metrics.
Coverage, member failures and exclusions remain explicit. Default prediction is
marginal with both endpoints untouched; known effects and support conditioning
require explicit regimes. See [the evaluation contract](docs/evaluation.md) and
run `python examples/evaluation.py` for a synthetic fitted example.
