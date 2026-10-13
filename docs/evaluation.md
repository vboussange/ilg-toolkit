# Predicting and evaluating a fold ensemble

A fold ensemble contains independently fitted `LandscapeGeneticsModel` members.
Each member composes a distance model with genetic calibration and retains the
observations used for training, validation-based selection and calibration.
Neural features are pooled across regions and species within a member; its MLPE
relationships and population effects are unpooled across region–species groups.
Members share an exact ordered species registry and common feature and target
contracts.

**Deployment prediction** uses every requested member at prepared locations.
**Out-of-fold (OOF) prediction** chooses contributors separately for each query
pair, using both nominal endpoint holdout and actual target access. **Scoring**
then compares those fixed predictions with measured query targets. Deployment
alone makes no held-out claim.

The [ensemble example](../examples/ensemble.py) demonstrates fitting, deployment,
OOF coverage, support conditioning and individual checkpoint continuation on
synthetic data. All groups use prepared `RegionBatch` inputs with species-keyed
`SamplingUnits`; observations remain separate. One biological sampling unit
belongs to one region within its species. The same textual population label in
different species does not identify a shared random effect or observation.

## Supply native region and species collections

Fold generation, ensemble fitting and OOF evaluation take a nonempty sequence of
regions, or a mapping keyed by each `RegionBatch.name`. Use this form for one
region too. Grouped observations, partitions, queries, supports and holdouts are
nested by region name, then species name. Missing observed species groups remain
absent; no complete region-by-species grid is required.

| Input | Native form |
| --- | --- |
| Regions for folds, fitting or evaluation | `[region]` or `{region.name: region}`; multiple regions use the same collection form |
| Observations for fitting or scoring | `{region.name: {species: PairwiseObservations(...)}}` |
| Fold `held_out_units` | `{region.name: {species: ("held-a", "held-b")}}` |
| Fold `training`, `validation`, `query` | Region–species mappings to `ObservationPartition` objects with matching embedded identity and role |
| OOF query `pairs` | Region–species mappings to nonempty labelled pair selections or matching query-role partitions |
| Convenience evaluation `partitions` | Region–species mappings to query-role partitions, or `None` for all measured pairs |
| Support-mode `support` | Region–species mappings to `EvaluationSupport` for exactly the queried groups |
| Generated-fold `holdout_size` | One integer, or a region–species integer mapping matching every observed group |

Every queried/support endpoint needs a location in its species' `SamplingUnits`.
The prepared feature meanings, order and count must match fitted members. All
observations use the identical `TargetSpec`: genetic meaning, kind, units and
transform. Mapping keys and embedded region/species identities must agree;
unknown species, malformed labels and incompatible contracts fail explicitly.

Operations already scoped to one region, such as `model.predict` and
`ensemble.predict`, take that `RegionBatch` directly. Their `pairs` argument is
species-keyed. Multispecies models require explicit `species=` selection; a sole
registered species may be inferred. Results remain species-keyed in both cases:

```python
# ensemble, region and query_pairs are already prepared; this region contains fox.
deployment = ensemble.predict(
    region, species="fox", pairs={"fox": query_pairs}
)["fox"]
print(deployment.prediction.values)  # Original genetic target units.
```

## Declare the study split

`generate_population_folds` draws seeded repeated holdouts independently within
each observed region–species group. It sorts group identities and unit labels
before drawing, so input presentation order does not change the folds. Draws may
overlap between folds; these are not disjoint K-fold partitions. Training uses
observed pairs with neither endpoint held out. By default, queries have both
endpoints held out; `query_regime="at_least_one_unseen"` also includes observed
pairs with one held-out endpoint. Missing edges remain absent. A draw without
observed training or query pairs fails explicitly.

Generated folds have `validation={}`. Query measurements are not automatically
reused for model selection. Supply explicit `PopulationFold` objects when the
study needs a validation split, spatial coordination or another partition policy.
MLPE still requires population sampling units and an identifiable training group;
the splitter does not make a statistically inadequate group valid.

For example, suppose one fox group contains four training populations, two
validation populations and two test populations. These sets are disjoint, and
all pairs shown below have observations:

```python
from itertools import combinations
from ilg_toolkit import ObservationPartition, PopulationFold

training_ids = ("train-a", "train-b", "train-c", "train-d")
validation_ids = ("validation-a", "validation-b")
test_ids = ("test-a", "test-b")
species = "fox"
fold = PopulationFold(
    fold_id="explicit-split",
    held_out_units={region.name: {species: validation_ids + test_ids}},
    training={region.name: {species: ObservationPartition(
        region.name, species, tuple(combinations(training_ids, 2)), role="training"
    )}},
    validation={region.name: {species: ObservationPartition(
        region.name, species, (validation_ids,), role="validation"
    )}},
    query={region.name: {species: ObservationPartition(
        region.name, species, (test_ids,), role="query"
    )}},
)
```

This design excludes every cross-partition pair. Training and calibration use
training pairs only; validation selects a snapshot using that group's frozen
training calibration. Test pairs are reserved for scoring fixed predictions.
Validation endpoints are nominally held out from training but become seen through
selection, so they cannot subsequently be claimed as untouched OOF endpoints.
If cross-partition pairs are part of the intended prediction question, declare
them explicitly and choose the corresponding endpoint regime; do not silently
mix that question with prediction between two unseen populations.

Automatic per-group draws do not establish geographic separation across species.
A pooled neural backbone can learn from another species in the same landscape.
For stronger spatial separation, assign geographic areas consistently across
species and construct coordinated folds with explicit training, validation and
query roles. Users choose the region boundaries and biological unit assignments.
Overlapping raster domains are allowed, but cropping changes graph paths and
spatial-model context; it is not a transparent equivalent of the full landscape.

A **common outer test set** is different from ordinary OOF aggregation. Reserve
that set before any member fitting or selection, restrict every member's
training/validation/calibration to development data, and verify the declared
access of every member. Then `ensemble.predict` evaluates the full deployed
ensemble on the common test pairs before their targets enter scoring. OOF may
instead use a different subset of members for each pair and may have no eligible
contributors. Those two evaluations answer different questions. Deployment does
not verify the outer-test design for the caller.

## Holdout and actual access must agree

`EvaluationRegime()` defaults to `prediction_mode="marginal"` and
`endpoint_regime="both_unseen"`. For each member and region–species group, a
qualifying endpoint must belong to both:

1. The member fold's nominal held-out sampling units.
2. Units untouched by actual distance-model training, validation selection and
   current MLPE calibration targets. Strict support policy also removes endpoints
   touched by declared support.

Both query endpoints must qualify by default. Set
`endpoint_regime="at_least_one_unseen"` to require one. It must be the **same
endpoint** satisfying holdout and non-access: a calibrated holdout cannot be
substituted by an unused nominal training endpoint.

The query pair's own target is forbidden under every regime, including known
effects and permissive support. Its use for training, selection, calibration or
support excludes that member. Reversing endpoints does not change identity.
Access is species-specific; a target in another species is not the same genetic
observation, although shared feature learning remains relevant to spatial design.

`OOFPrediction.access[member_id][region_name][species_name]` records actual pair
sets and calibration roles. An MLPE head can list a held-out population with a
prior-only effect; `population_ids` membership alone does not mean its targets
were accessed. Explicit recalibration changes eligibility according to its saved
roles and pairs, even when the nominal fold remains unchanged. Assigning a query
role does not erase earlier access.

Fitting records training and selection access. A manually constructed model with
an entirely empty `training_pairs={}` has unknown distance-model access and is
excluded conservatively. A genuinely untrained group can truthfully appear as
`training_pairs={region.name: {species: ()}}`. A complete declared access manifest
may also show no consumed targets in a particular query group. These declarations
must reflect actual history; validation and calibration access remain relevant.

## Predict first, score afterward

The following calls assume a fitted ensemble, prepared query region, a nonempty
pair selection and measured `PairwiseObservations` named `observed`:

```python
from ilg_toolkit import predict_out_of_fold, score_out_of_fold

regions = [region]
queries = {region.name: {"fox": query_pairs}}
observations = {region.name: {"fox": observed}}
prediction = predict_out_of_fold(ensemble, regions, queries)
# Prediction is fixed before query measurements enter scoring.
result = score_out_of_fold(prediction, observations)
print(result.n_pairs, result.n_query_pairs, result.coverage)
print(result.by_group[region.name]["fox"].rmse)
print(result.by_species["fox"].coverage)
```

`predict_out_of_fold` accepts no query observations or target values. It computes
only eligible predictions: an unrelated ineligible pair cannot first fail its
genetic inverse transform. Changing scoring measurements changes errors, not
this prediction object's values, access snapshots or eligibility.

`evaluate_ensemble(ensemble, regions, observations, partitions=query_partitions)`
performs prediction then scoring. Its optional nested partitions must have role
`"query"`, match region/species identities and select observed pairs. Omitting
them requests every measured pair, including pairs with zero eligibility. The
convenience call passes only identities and target metadata into prediction;
measurements enter scoring afterward. Missing measurements remain absent.

OOF queries canonicalize and deduplicate reversed or repeated pairs within each
group. Flat keys are `(region_name, species_name, (left, right))`, sorted by region,
species and canonical unordered pair. Ordinary model/deployment prediction
preserves caller order and orientation and **rejects** duplicates, including
reversals. Omitting ordinary `pairs` requests all distinct pairs for each selected
species in its sampling-unit order. All operations reject self-pairs and unknown
or malformed labels.

## Read predictions, coverage and metrics

A common `Prediction` stores original-scale `values` aligned with `pairs`,
`region_name`, `species_name`, `target` and `sampling_unit_ids`. Conditional and
deployment ensemble results compose this record as `.prediction`. OOF stores
`.predictions[region_name][species_name]`; its flat `.keys` and `.values` derive
from those group records.

`prediction.as_matrix(order=None)` represents stored points as a symmetric
matrix. Model operations retain the full query group's sampling-unit order;
standalone MLPE defaults to distinct query endpoints in first-appearance order.
An explicit order must be unique and include every predicted endpoint; it can
include other units. Unrequested off-diagonals are NaN. Structural zero diagonals
are a dissimilarity convention, not fitted self-predictions; conversion refuses
other target kinds without that convention. It adds no points or covariance.

For `k` requested members and `m` unique region–species queries:

| `OOFPrediction` field | Shape or meaning |
| --- | --- |
| `member_ids`, `member_statuses` | Complete requested member order and fit statuses, including failed/pending members |
| `member_values` | JAX `(k, m)` original-scale points; NaN when excluded or failed |
| `eligible_counts`, `covered_mask` | Host NumPy `(m,)` successful finite contributor counts and `count > 0` mask |
| `eligible_member_ids` | Successful contributor identities per query, in member order |
| `values`, `member_spread` | JAX `(m,)` mean and descriptive population SD (ddof=0); NaN without contributors |
| `coverage`, `coverage_status` | Covered fraction and `"complete"`, `"partial"` or `"none"` |
| `exclusion_reasons` | Member mapping to `m` reasons in key order; `None` denotes success |
| `member_failures` | Failed-fit and pending-member diagnostics |
| `prediction_failures` | Member → region → species prediction diagnostics; fit status remains unchanged |
| `access`, `conditioning_provenance` | Member → region → species actual-use snapshots and successful conditional provenance |
| `member_model_variances` | `None` in marginal mode; otherwise JAX `(k, m)` model-scale variances, NaN where unavailable |

Counts record successful contributions, not merely nominal eligibility. A failed
prediction does not increase them. Reasons distinguish unknown access, own-query
access, absent group holdout, insufficient unseen endpoints, unavailable members
and prediction failure. Every requested member stays visible; another member's
failure need not prevent complete pair coverage.

Deployment requires all requested members to complete and predict successfully;
it refuses to shrink the ensemble silently. OOF retains failures while allowing
other members/groups to contribute. If every member is unavailable, label-free
prediction requires explicit `target=TargetSpec(...)` metadata; convenience
evaluation obtains it from observations. Invalid forms, roles, identities,
contracts or support declarations raise `ValueError`. Member-level numerical,
unsupported-mode or provenance failures are recorded by OOF; nonfinite aggregate
or scoring arithmetic raises `FloatingPointError`.

Each member's point is inverse-transformed into original genetic units **before**
equal-member averaging. `aggregate_predictions` checks region, species, target,
scale, sampling-unit order and exact pair order. Scoring then counts each covered
unique group/pair **once**, regardless of contributor count. It needs a measured
target for every requested query, including uncovered ones; additional measured
pairs do not contribute errors.

`OOFEvaluation` exposes overall counts/errors, `.by_group[region][species]` and
`.by_species[species]`. Each group/species `EvaluationMetrics` record has:

| Field | Meaning |
| --- | --- |
| `n_pairs` | Covered unique pairs scored |
| `n_query_pairs` | All requested unique pairs, covered or not |
| `coverage` | `n_pairs / n_query_pairs` |
| `mse` | Mean squared error in squared original target units |
| `rmse`, `mae` | Root mean squared error and mean absolute error in original units |

Queried groups/species with no coverage remain present and all three errors are
`None`. Missing predictions never imply zero error. `observed_values` follows the
prediction's flat key order. Overall and per-species errors give every covered
pair equal weight, so groups with more covered pairs contribute more weight.
This is separate from training's equal-species, then equal-observed-region
objective. Read coverage alongside each error; different covered subsets are not
automatically comparable estimates of the same test population.

Conductance summaries and ecological distances are separate diagnostics:
`ensemble.conductance_surfaces(region, species=...)` returns species-keyed
`EnsembleSurface` records, while `ensemble.ecological_distances(region, species=...)`
returns species → member ID → full matrix. Mean conductance does not determine
mean resistance, and mean ecological distance does not determine mean calibrated
genetic prediction. Embedding models do not supply conductance surfaces.

## Explicit known effects and support

Marginal MLPE prediction uses zero population-effect means. Known effects use the
saved group's posterior only when explicitly requested:

```python
from ilg_toolkit import EvaluationRegime

regime = EvaluationRegime(
    endpoint_regime="at_least_one_unseen", prediction_mode="known_effects"
)
prediction = predict_out_of_fold(ensemble, regions, queries, regime=regime)
```

An unseen population receives an independent prior effect. Known effects do not
relax endpoint or own-query rules, and direct calibration has no MLPE posterior.
Genetic MLPE prediction requires a calibrated region–species group, even when
raw ecological distances are already available there.

Support consumes only explicitly declared measurements:

```python
from ilg_toolkit import EvaluationSupport, ObservationPartition

# support_observed contains only disjoint support targets for this fox group.
declared = ObservationPartition(
    region.name, "fox", support_observed.observed_pairs, role="support"
)
support = {region.name: {"fox": EvaluationSupport(support_observed, declared)}}
regime = EvaluationRegime(
    prediction_mode="support", support_endpoint_policy="allow_declared_support"
)
prediction = predict_out_of_fold(
    ensemble, regions, queries, regime=regime, support=support
)
result = score_out_of_fold(prediction, observations)
```

Support declarations cover exactly the queried groups. Their role, region,
species and target match, and measured support identities equal the declaration.
Support must be disjoint from **all** requested query identities, even uncovered
ones. Each member also rejects support reusing its current calibration targets.
Conditional output identity, mode, scale and calibration/support role provenance
must match the declared access before that member contributes. Support outside
support mode, or omitted from support mode, fails explicitly.

The default `support_endpoint_policy="unseen"` treats support endpoints as seen.
`"allow_declared_support"` checks endpoint eligibility against
training/selection/calibration before consuming additional disjoint support.
It evaluates explicitly conditioned predictions; those endpoints are no longer
untouched afterward. This option is valid only in support mode and never permits
the query's own target.

For deployment without OOF restrictions, use
`model.predict_known_effects(region, species=..., pairs={...})` or
`model.predict_with_support(region, support_observations, species=..., pairs={...},
support_partition={...})`. Here support observations/partitions and query pairs
are species-keyed and results are species-keyed conditional records. These calls
freeze the entire distance model, coefficients and variance parameters; they
return conditional predictions without mutating saved population posteriors.
See [conditioning](mlpe-conditioning.md) for the Gaussian calculation and guards.

## Continue a member after a completed update

`fit_ensemble` fits independent fold/initialization-seed members using an explicit
`model_factory(key)` and `TrainingConfig`. The factory receives the declared
initialization key. A resumed member uses its saved model, optimizer and random
state; its factory is not invoked. `on_step(identity, state)` runs after each
completed optimizer update, including steps without full evaluation. It does not
run at initialization or when continuation has no updates left. Callback
exceptions propagate and stop later member work; ordinary initialization/fit
failures remain explicit member outcomes. `on_member(member)` reports each
returned member outcome separately, including failures.

Checkpoint filenames and member orchestration remain caller-owned. The following
pattern assumes compatible prepared `regions`, nested `observations`, `fold`
and an explicit native `model_factory`. Enable x64 before constructing MLPE inputs:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
from ilg_toolkit import (
    TrainingConfig, fit_ensemble_member, load_checkpoint, save_checkpoint,
)

class PauseMember(RuntimeError):
    pass

config = TrainingConfig(objective="mlpe", steps=2, evaluation_interval=2)
with TemporaryDirectory() as directory:
    path = Path(directory) / "member.ilg"

    def checkpoint_and_pause(identity, state):
        if state.step == 1:
            save_checkpoint(path, state)
            raise PauseMember(identity.member_id)

    try:
        fit_ensemble_member(
            regions, observations, fold=fold, initialization_seed=17,
            config=config, model_factory=model_factory,
            on_step=checkpoint_and_pause,
        )
    except PauseMember:
        pass

    template = model_factory(jax.random.PRNGKey(0))
    state = load_checkpoint(path, distance_model=template)
    resumed = fit_ensemble_member(
        regions, observations, fold=fold, initialization_seed=17, state=state,
    )
```

The saved configuration still requests two steps. At interruption, `.step == 1`
while history contains only the initial full evaluation; the latest parameters
and optimizer state differ from the independently selected calibrated snapshot.
Resume completes the same original budget, including its final evaluation. Do not
shorten a first run's budget to simulate interruption: that introduces another
mandatory final evaluation. Exact continuation uses unchanged inputs, policy,
precision and execution conditions. Loading requires the exact species registry,
architecture, static configuration and dtype template; only its initial learned
values may differ. See [persistence](persistence.md) for schema and integrity rules.
`fit_ensemble(..., member_states={member_id: state}, ...)` applies the same rule to
caller-managed collections of member checkpoints.

## Keep scales, uncertainty and precision distinct

Conditional `model_values` are transformed-scale means; `prediction.values` are
their inverse-transformed points. Under a nonlinear inverse these are not
original-scale expectations; no bias correction is applied. Signed MLPE points
are not clipped.

Conditional predictive variance includes population-effect posterior uncertainty,
independent residual variance and configured jitter in **squared transformed
model units**. Distance parameters, fixed coefficients and variance parameters
are held fixed, so their uncertainty is excluded. OOF retains these variances
per member and constructs no aggregate statistical variance. `member_spread`
describes disagreement in original target units; it is neither conditional
variance nor a confidence interval. One contributor gives zero descriptive
spread; no contributors gives NaN. Marginal variance is unreported (`None`), not
zero. Separate pair variances do not constitute a joint covariance matrix.

Precision depends on the operation:

- Ordinary/direct predictions and OOF output arrays follow configured JAX
  precision, including float32 workflows. Scoring conversion/arithmetic follow
  that policy; metric summaries are Python floats.
- Standalone MLPE calibration/marginal prediction, conditional posterior
  calculations and ensemble aggregation use scoped float64. Enclosing OOF arrays
  retain configured precision when storing those member outputs.
- Effective resistance and joint MLPE training require explicitly enabled x64.
  Enable it before constructing measurements; later promotion cannot restore
  information already rounded away.
- Matrix conversion and OOF flat views retain stored precision, including
  standalone float64 points after their scope ends. No operation silently changes
  global JAX precision. Labels/provenance and NumPy coverage metadata stay on the host.
