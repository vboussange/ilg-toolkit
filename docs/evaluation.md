# Evaluate fold ensembles

This guide covers [fitting ensemble members](#fit-an-ensemble),
[prediction and scoring](#predict-then-score), [holdout rules](#holdouts-and-observation-use),
[coverage and uncertainty](#read-coverage-and-uncertainty), and
[support conditioning](#condition-on-support). The final section describes
[experimental multispecies inputs](#experimental-multispecies-evaluation).
The [ensemble example](../examples/ensemble.py) runs a single-species workflow.

## Fit an ensemble

A fold ensemble contains landscape genetics models fitted with different
sampling-unit holdouts and initialization seeds. `fit_ensemble` accepts region
batches, observations, a `model_factory(key)`, and a `TrainingConfig`, together
with `n_folds`, `holdout_size`, `fold_seed`, and `initialization_seeds`. Training
pairs join sampling units outside each fold's holdout. `PopulationFold` records
training, validation, query, and held-out sampling-unit identities.

Each ensemble member retains its fitted model and observation-use records.
`fit_ensemble_member` supports fitting or continuing one member; see
[persistence](persistence.md) for checkpoint handling. Member failures remain
available as `ensemble.failures`.

## Predict, then score

Deployment prediction, `ensemble.predict(region, pairs=...)`, averages every
member's genetic predictions on the original scale. Its `EnsemblePrediction`
contains the mean as `.prediction`, individual values as `.member_values`, and
descriptive standard deviation as `.member_spread`.

Out-of-fold (OOF) prediction chooses eligible members separately for each query
pair. Supply a fitted `ensemble`, a region batch, and query-pair identifiers:

```python
from ilg_toolkit import predict_out_of_fold, score_out_of_fold

prediction = predict_out_of_fold(ensemble, region, query_pairs)
result = score_out_of_fold(prediction, observations)
print(result.n_pairs, result.n_query_pairs, result.coverage)
print(result.mse, result.rmse, result.mae)
```

Query measurements enter the scoring call after predictions are fixed.
`evaluate_ensemble(ensemble, region, observations, partitions=query_partition)`
combines these operations. Its optional partition has role `"query"`; omitting it
requests all measured pairs.

For several regions, use a sequence of region batches or a name-keyed mapping.
`predict_out_of_fold` takes a mapping from each query region to its pairs, and
`score_out_of_fold` takes observations keyed by those same regions. Every queried
sampling unit needs a location, and features and targets must match the fitted
models. OOF queries are deduplicated, treating reversed pairs as the same query,
and sorted by region and sampling-unit identifiers.

## Holdouts and observation use

The default `EvaluationRegime(endpoint_regime="both_unseen")` requires both
sampling units in a query pair to meet two conditions for a member:

1. They belong to that member's fold holdout.
2. Their genetic observations were unused for training, validation-based model
   selection, calibration, or support conditioning.

`endpoint_regime="at_least_one_unseen"` requires one sampling unit to satisfy
both conditions. A member is excluded whenever it used the query pair's own
measurement, under every prediction mode. Observation identity includes region
and species; pair reversal preserves identity.

Fitting records observation use automatically. For manually constructed models,
empty `training_pairs={}` means unknown training use and causes exclusion. An
untrained model can record `training_pairs={region.name: ()}` when that accurately
describes its history. Calibration's population list may include sampling units
with prior-only effects; eligibility follows the pairs used for fitting.

Note: Deployment predictions carry no holdout guarantee.

## Read coverage and uncertainty

`OOFPrediction.predictions[region.name]` contains each region's `Prediction`.
Its `keys` order aligns the combined arrays:

| Result | Meaning |
| --- | --- |
| `values`, `member_values` | Original-scale means and values from each requested member |
| `eligible_counts`, `eligible_member_ids` | Successful contributors for each query pair |
| `covered_mask`, `coverage`, `coverage_status` | Covered pairs, covered fraction, and `"complete"`, `"partial"`, or `"none"` |
| `exclusion_reasons`, `access` | Eligibility decisions and observation use |
| `member_failures`, `prediction_failures` | Fitting and prediction failures |
| `member_spread` | Descriptive standard deviation across successful members |
| `member_model_variances` | Per-member conditional variances on the model scale, or `None` in marginal mode |

Failed predictions contribute zero to coverage counts. Uncovered pairs have
not-a-number (`NaN`) values. If all members are unavailable, `predict_out_of_fold` requires
`target=TargetSpec(...)`; `evaluate_ensemble` obtains the target from observations.
Deployment propagates a member failure, while OOF prediction can retain partial
coverage from successful members.

Scoring requires a measurement for every requested pair, including uncovered
pairs. Each covered pair contributes once after averaging successful member
predictions. Mean squared error (`mse`) has squared target units; root mean
squared error (`rmse`) and mean absolute error (`mae`) have original target units.
At zero coverage, all three metrics are `None`.

Conditional variances describe population-effect and observation uncertainty
with model parameters fixed. See [MLPE conditioning](statistics.md#mlpe-conditioning),
where MLPE means maximum-likelihood population effects, for their interpretation
and precision requirements. `member_spread` describes disagreement in original
units.

Note: Member spread is not a confidence interval. Conditional variances stay on
the model scale and are not combined into an ensemble statistical variance.

## Condition on support

`EvaluationRegime(prediction_mode="known_effects")` uses saved population effects.
Support conditioning uses an `EvaluationSupport` containing observations and a
matching `ObservationPartition` with role `"support"`:

```python
from ilg_toolkit import EvaluationRegime, EvaluationSupport

support = EvaluationSupport(support_observations, support_partition)
regime = EvaluationRegime(
    prediction_mode="support", support_endpoint_policy="allow_declared_support"
)
prediction = predict_out_of_fold(
    ensemble, region, query_pairs, regime=regime, support=support,
)
result = score_out_of_fold(prediction, observations)
```

Support identities must match the support partition and be disjoint from all
query identities. Each member also checks that support pairs are disjoint from
its calibration pairs. Conditional modes require MLPE calibration for the region.

The default `support_endpoint_policy="unseen"` counts sampling units with support
observations as seen. `"allow_declared_support"` checks holdout eligibility before
consuming disjoint support and evaluates the resulting conditional prediction.
Support is required in support mode and rejected in other modes.

## Experimental multispecies evaluation

`fit_ensemble(..., species=order)` retains the model's species order. Sampling-unit
holdouts, observation use, and eligibility are assessed per region–species group.
Repeated sampling-unit identifiers in different species remain separate.

`ensemble.predict` and `predict_out_of_fold` accept `species="name"` or a sequence
of names. Deployment returns one result for a string and a mapping for a sequence.
For named species, OOF pair and support mappings use `(region_name, species_name)`
keys, as do `predictions` and `access` records. A single queried group can use a
pair sequence directly. Flat query order is region, species, then pair.

Scoring accepts a species-labeled observations sequence for one region, or a
mapping from region names to those sequences. Each covered region–species pair
contributes once to pooled metrics. `by_species` adds metrics when the ensemble
models more than one species. Only groups present in the queried regions are
required, supporting species with different regional coverage.
