# Predicting and evaluating a fold ensemble

A fold ensemble contains independent fitted `LandscapeGeneticsModel` members.
Each member composes a distance model with a genetic calibration module and
retains the targets used for training, validation-based selection and calibration.
A population fold declares which sampling units were held out. Genetic
observations are unordered pairs of those units; identical labels in different
regions identify different pairs.

**Deployment prediction** uses every requested member to predict genetic points
at prepared locations. It makes no claim that endpoints or targets were held
out. **Out-of-fold (OOF) prediction** selects contributors separately for each
query pair, using both nominal endpoint holdout and the member's actual target
access. **Scoring** subsequently compares those fixed points with measured query
targets. Keeping these steps separate makes the information boundary explicit.

Run the complete synthetic workflow from the repository root:

```sh
JAX_PLATFORMS=cpu python examples/ensemble.py
```

The script creates all features, locations and measurements, fits two independent
MLPE members, compares deployment with eligible OOF prediction, prints coverage,
and conditions on declared support before scoring separate queries. It enables
x64 in an explicit scope and needs no study data. Its final section demonstrates
caller-owned individual checkpoint continuation.

## Choose what information prediction may use

A query pair names endpoints whose target is to be predicted. A support pair
provides a measured target explicitly allowed to condition population effects.
Training, validation-selection and calibration pairs have already supplied
information to a member. Giving a pair a query role does not erase earlier access.

| Operation | Information used | Holdout requirement | Result |
| --- | --- | --- | --- |
| `ensemble.predict(region, pairs=None)` | Every member's saved marginal calibration; MLPE effect means are zero | None | `EnsemblePrediction`: original-scale mean points, all member points and descriptive SD |
| `model.predict(region, pairs=pairs, known_effects=True)` | Saved regional MLPE effect posterior | None for this deployment call | `Prediction`: original-scale values and model-scale values/variances in `extras` |
| `model.predict(region, pairs=pairs, support=observations)` | Saved population effects conditioned on support observations | None for this deployment call; support/query identities must be disjoint | `Prediction`: original-scale values and model-scale values/variances in `extras` |
| `predict_out_of_fold(ensemble, region, pairs, ...)` | Only eligible members; marginal by default, conditional only by explicit regime | Endpoint and actual-access rules below; own-query target always forbidden | Regional points, member outputs, coverage, access and visible failures |
| `score_out_of_fold(prediction, observations)` | Measured query targets after prediction is fixed | Uses the supplied prediction's eligibility and coverage | Covered-pair MSE, RMSE and MAE |
| `evaluate_ensemble(ensemble, region, observations, ...)` | Extracts identities/metadata, predicts without query measurements, then scores | Same OOF rules | `OOFEvaluation` convenience result |

These operations do not refit or mutate models. Support conditioning returns a
conditional result without altering the saved posterior. Deployment requires
all requested members to succeed; OOF can retain partial coverage with failures
reported. MLPE genetic prediction requires explicit calibration for the query
region even when the distance model can already compute its ecological distances.

## Holdout and actual access must agree

`EvaluationRegime()` defaults to `prediction_mode="marginal"` and
`endpoint_regime="both_unseen"`. For each member and region, a qualifying endpoint
must be in both of these sets:

1. The member fold's nominal held-out sampling units.
2. Units untouched by actual distance-model training, validation-based selection
   and regional MLPE calibration targets; strict support policy also removes
   endpoints touched by declared support.

Both query endpoints must qualify by default. Set
`endpoint_regime="at_least_one_unseen"` to require one. It must be the **same
endpoint** that satisfies holdout and non-access: a calibrated holdout cannot be
substituted by a nominal training endpoint that happens to be unused.

The query pair's own target is forbidden under every regime, including
`at_least_one_unseen`, known effects and permissive support. Its use for training,
validation selection, calibration or support excludes that member. Pair reversal
does not change identity. `OOFPrediction.access[member_id][region_name]` records
these actual pair sets and calibration roles. A regional MLPE fit can list a
held-out population with a prior-only effect; membership in `population_ids`
alone does not mean its targets were accessed. Fitting calibration can change
eligibility because the current regional fit's calibration access is retained.

Fitting records training and selection access automatically. A manually
constructed model with entirely empty `training_pairs={}` has unknown
distance-model access and is excluded conservatively. A genuinely untrained
model can truthfully declare `training_pairs={region.name: ()}`. This declaration
must describe actual history; validation and calibration access must also remain
truthful.

## Predict first, score afterward

The following calls assume fitted `ensemble`, prepared `region`, a nonempty
labelled `query_pairs` selection, and measured `observations`:

```python
from ilg_toolkit import predict_out_of_fold, score_out_of_fold

prediction = predict_out_of_fold(ensemble, region, query_pairs)
# Prediction is now fixed. Query measurements enter only this scoring call.
result = score_out_of_fold(prediction, observations)
print(result.n_pairs, result.n_query_pairs, result.coverage)
print(result.mse, result.rmse, result.mae)
```

`predict_out_of_fold` takes no query observations or target values. It predicts
only eligible pairs: an unrelated ineligible pair cannot first fail its genetic
inverse transform. Changing scoring measurements can change errors but cannot
change this prediction object, its access snapshots or eligibility.

For convenience, `evaluate_ensemble(ensemble, region, observations,
partitions=query_partition)` performs the two steps. Its optional partitions
must have role `"query"`, match region names and select observed pairs. Omitting
partitions requests every measured pair, including pairs with zero eligibility.
Missing measurements stay absent. The convenience function reads measurements
for scoring and target metadata, but passes only identities and metadata to the
prediction stage.

### Single and shared regions

| Input | One region | Shared regions |
| --- | --- | --- |
| `region` | `RegionBatch` | Nonempty sequence with unique names, or mapping keyed by each batch's name |
| OOF `pairs` | Nonempty labelled pair iterable or matching query-role `ObservationPartition` | Mapping with exactly every query region name, each containing one such selection |
| Scoring `observations` | `PairwiseObservations` | Mapping with exactly the prediction region names |
| Convenience evaluation `observations` | `PairwiseObservations` | Aligned sequence in supplied region order or name-matching mapping |
| Convenience evaluation `partitions` | Query-role partition, or `None` for all measured pairs | Aligned sequence/name-matching mapping, or `None` |
| Support-mode `support` | `EvaluationSupport` | Mapping declaring every query region exactly |

Every queried/support endpoint needs a location in its prepared region. Feature
meanings/counts must match fitted members. All regions and scoring/support
observations must share the identical `TargetSpec`: genetic meaning, kind, units
and fitted transform. Name-matching mappings have exact key sets; extra or
missing regions fail. Each query region needs at least one pair.

OOF evaluation canonicalizes and deduplicates queries, including reversed or
repeated input pairs. Its flat order is sorted region name, then
lexicographically sorted endpoint labels and unordered pair identities. Ordinary
`model.predict` and `ensemble.predict` preserve explicit caller order/orientation
and **reject** duplicates, including reversals. Their omitted `pairs` requests
all distinct pairs in region sampling-unit order. All operations reject self-pairs
and unknown or malformed labels.

## Read points, coverage and metrics

Ordinary results use `Prediction`: `values` has shape `(m,)`, aligned with `pairs`,
`region_name`, optional `species`, and `sampling_unit_ids`. Named `extras` contain
arrays aligned with those same pairs. Conditional and deployment ensemble
results compose this as `.prediction`. Out-of-fold results use
`.predictions[region_name]` for each regional record and retain the target on the
`OOFPrediction` for scoring.

`prediction.as_matrix(name=None)` rearranges stored values, or a named extra,
into a symmetric matrix in the record's sampling-unit order. Both the diagonal
and unrequested pairs are NaN. Model results retain the query region's order;
standalone MLPE results use query sampling units in first-appearance order.

For `k` requested members and `m` unique regional queries, `OOFPrediction` exposes:

| Field | Shape / meaning |
| --- | --- |
| `member_ids`, `member_statuses` | Complete requested order and fitting statuses, including failed/pending members |
| `member_values` | JAX `(k, m)` original-scale points; NaN where excluded or failed |
| `eligible_counts`, `covered_mask` | Host NumPy `(m,)` successful finite contributor counts and `count > 0` mask |
| `eligible_member_ids` | One tuple per unique query, naming successful contributors in member order |
| `values`, `member_spread` | JAX `(m,)` mean points and descriptive population SD (ddof=0); NaN with no contributors |
| `coverage`, `coverage_status` | Covered fraction and `"complete"`, `"partial"` or `"none"` |
| `exclusion_reasons` | Member mapping to `m` reasons; `None` means successful contribution |
| `member_failures` | Failed-fit and pending-member diagnostics |
| `prediction_failures` | Member/region prediction diagnostics; completed fitting status is unchanged |
| `access` | Training, validation, calibration and support observation use by member and region |
| `member_model_variances` | `None` in marginal mode; conditional JAX `(k, m)` model-scale variance, NaN when unavailable |

Counts refer to successful contributions, so a nominally eligible member with a
prediction failure does not increase them. Exclusion reasons distinguish unknown
access, own-query access, absent regional holdout, insufficient unseen holdouts,
unavailable members and prediction failure. Every requested member stays visible;
coverage can remain complete even if another member fails.

If every member is unavailable, label-free prediction requires explicit
`target=TargetSpec(...)` metadata to report uncovered points. `evaluate_ensemble`
obtains it from observations. Invalid input forms, roles, labels, target contracts
or support declarations raise `ValueError`. OOF records member-level numerical,
unsupported-mode and provenance failures, allowing other members to contribute;
deployment propagates failures. Nonfinite aggregate or metric arithmetic raises
`FloatingPointError`.

Scoring requires a measured original-scale target for **every** requested query,
even uncovered ones; extra measurements are ignored. Several successful member
points are averaged after each member's inverse transformation, then each
covered unique region/pair contributes **once** to pooled metrics. Every covered
pair has equal weight regardless of contributor count or region. This is separate
from equal-region weighting during training.

`OOFEvaluation.observed_values` is `(m,)` in key order. `n_pairs` counts covered
unique pairs, `n_query_pairs` counts all requested pairs, and `coverage` is their
ratio. MSE has squared target units; RMSE and MAE have original units. Uncovered
pairs do not contribute errors. At zero coverage, all three metrics are `None`:
missing predictions never imply zero error.

## Explicit known effects and support

Known-effect OOF uses the saved regional posterior only when explicitly requested:

```python
from ilg_toolkit import EvaluationRegime

regime = EvaluationRegime(
    endpoint_regime="at_least_one_unseen", prediction_mode="known_effects"
)
prediction = predict_out_of_fold(ensemble, region, query_pairs, regime=regime)
```

Stored calibration access still applies. An unseen population receives an
independent prior effect; known effects do not silently relax endpoint or
own-query rules. Direct calibration does not supply an MLPE effect posterior.

Support conditioning consumes only an explicit declaration:

```python
from ilg_toolkit import EvaluationSupport

support = EvaluationSupport(support_observations, support_partition)
regime = EvaluationRegime(
    prediction_mode="support", support_endpoint_policy="allow_declared_support"
)
prediction = predict_out_of_fold(
    ensemble, region, query_pairs, regime=regime, support=support
)
result = score_out_of_fold(prediction, observations)
```

`support_partition` must have role `"support"` and the matching region name;
measured support identities must equal it exactly. Support must be disjoint from
**all** requested query identities, not only covered ones. Each member also
rejects support that reuses its current calibration targets. Conditional access
provenance must agree with the declared mode and pair/role sets before contributing.
Supplying support outside support mode or omitting support in that mode fails.

The default `support_endpoint_policy="unseen"` counts support endpoints as seen,
so strict untouched queries can lose coverage. `"allow_declared_support"` checks
endpoint holdout against training/selection/calibration **before** consuming
additional disjoint support. It explicitly evaluates conditional prediction and
does not claim those endpoints remain untouched afterward. It is valid only in
support mode, and the query's own target remains forbidden.

## Keep scales, uncertainty and precision distinct

Conditional `model_values` are transformed-scale means; `prediction.values` are
their inverse-transformed points. A nonlinear inverse does not make these points
original-scale expectations; there is no bias correction. Saved marginal MLPE
points use zero population-effect means. Signed MLPE points are not clipped.

Conditional predictive variance includes population-effect posterior uncertainty,
independent residual variance and configured jitter, in **squared transformed
model units**. Distance-model parameters, fixed coefficients and variance
parameters are held fixed, so their uncertainty is excluded. OOF retains these
variances per member and never manufactures an aggregate statistical variance.
`member_spread` describes member disagreement in original target units; it is
neither conditional variance nor a confidence interval. Marginal variance is
unreported (`None`), not zero. No operation silently inverse-transforms variance
into original-scale Gaussian uncertainty or interprets pair variances as a joint
covariance.

Precision depends on the operation:

- Ordinary/direct predictions and OOF output arrays follow configured JAX
  precision, including supported float32 workflows. Scoring target conversion
  and metric arithmetic also follow that policy; metrics are returned as Python
  floats. Configure sufficient precision before creating input measurements.
- Standalone MLPE calibration/marginal prediction, conditional posterior
  calculations and ensemble aggregation use scoped float64. Enclosing OOF arrays
  retain configured precision when storing those member outputs.
- Effective resistance calculations and joint MLPE training require explicitly
  enabled x64; enable it before those operations. Scoped later promotion cannot
  restore precision already lost when constructing observations.
- Matrix conversion and OOF flat views retain stored array precision, including
  standalone float64 points after their scope ends. These operations do not
  silently change global JAX precision. Labels/provenance remain host records;
  coverage counts and masks are host NumPy metadata.
