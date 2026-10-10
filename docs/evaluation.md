# Eligibility-aware ensemble evaluation

Deployment `ensemble.predict(region)` averages all members. Out-of-fold
prediction selects members separately for each canonical `(region, unordered
pair)` and reports coverage. Run `python examples/ensemble.py` for the complete
synthetic fit and prediction workflow.

| Operation | Genetic information used for prediction | Evaluation requirement | Returned quantities |
| --- | --- | --- | --- |
| `ensemble.predict(region)` | Each member's saved calibration; population effects marginalised | Deployment; all requested members must be available | Original-scale mean, member values and descriptive SD |
| `model.predict_known_effects(region, pairs)` | Saved posterior population effects | Explicit mode; accessed calibration targets cannot be OOF queries | Original-scale points, model-scale variance and access provenance |
| `model.predict_with_support(...)` | Saved posterior plus exactly declared disjoint support targets | Explicit support partition; query targets remain separate | Conditional points, model-scale variance and support provenance |
| `predict_out_of_fold(...)` | Only eligible members' marginal predictions by default | Nominal holdout and actual unseen endpoint access; own-query target forbidden | Unique-pair predictions, eligible counts, coverage and descriptive SD |
| OOF with an explicit known/support `EvaluationRegime` | Stored effects or declared support, as selected | Same own-target rules; declared support endpoint policy retained | Conditional OOF points/coverage and separate per-member model variances |
| `score_out_of_fold(prediction, observations)` | Reads original-scale query targets after predictions and eligibility are frozen | Scores each covered unique pair once | MSE, RMSE, MAE, covered-pair count and observed targets; no uncertainty interval |

Numerical means, member values, spreads and conditional variances are JAX arrays.
Coverage masks/counts and access/provenance records retain labelled host metadata.

```python
from ilg_toolkit import evaluate_ensemble

result = evaluate_ensemble(ensemble, region, observations, partitions=query_partition)
print(result.n_pairs, result.n_query_pairs, result.coverage)
print(result.mse, result.rmse, result.mae)
print(result.predictions.eligible_counts)
```

`partitions` accepts query-role partitions aligned with the single/shared regional
input contract. Omit it to consider every measured pair: training and other
ineligible pairs remain present with zero coverage. Missing observations stay
absent. Canonical query identities are unique, including reversed duplicates
supplied to the label-free API. Several eligible predictions are averaged before
a covered pair contributes **once** to pooled metrics. Pooled metrics weight each
covered pair equally, independently of the equal-region training loss.

## Predict before reading query targets

```python
from ilg_toolkit import predict_out_of_fold, score_out_of_fold

prediction = predict_out_of_fold(ensemble, region, [("population-a", "population-b")])
result = score_out_of_fold(prediction, observations)
```

The prediction operation accepts prepared locations and labelled pairs without
query genetic observations. For several regions, supply prepared regions and a
pair mapping keyed by region name. It predicts only eligible pairs; unrelated
ineligible predictions cannot fail an inverse transformation first. Scoring reads
original-scale targets afterward. Changing those targets can change metrics but
cannot change previously generated predictions or eligibility.

`OOFPrediction` retains canonical `keys`, original-scale `values`, `target`,
`member_ids`, `member_values`, `eligible_member_ids`, `eligible_counts`,
`covered_mask`, `coverage`, `coverage_status`, and descriptive `member_spread`
(population SD, ddof=0). Uncovered entries are NaN with zero counts. With no
coverage, metrics are `None`; zero error is never invented. MSE has squared target
units; RMSE, MAE and descriptive member SD have original target units.

All requested members remain represented. `member_statuses`, `member_failures`,
`prediction_failures` and per-member `exclusion_reasons` expose unavailable fits,
pending members, numerical prediction errors and information-access exclusions.
Other eligible members can still supply coverage, with these failures visible.
If every member is unavailable, label-free prediction requires explicit
`target=TargetSpec(...)`; the evaluation wrapper obtains that metadata from the
observations. Incompatible target contracts fail clearly.

## Endpoint holdout and actual target access

`EvaluationRegime()` defaults to marginal prediction and `both_unseen`.
Eligibility counts query endpoints in the intersection of:

1. The member fold's nominal held-out sampling units for this region.
2. Endpoints untouched by that member's actual encoder training, validation-based
   selection and genetic calibration (and support, under the strict policy).

Both endpoints must qualify by default. Set
`EvaluationRegime(endpoint_regime="at_least_one_unseen")` to require one qualifying
endpoint. It must be the **same** endpoint that is held out and remains unseen.
A recalibrated held-out endpoint cannot be replaced by an unused nominal training
endpoint. The query pair's own target is forbidden under every regime, including
validation selection and calibration access, even when the other endpoint would
otherwise qualify. Region names are part of identity when labels repeat.

The `access` snapshots retain training, validation, calibration and support pairs.
MLPE `head.population_ids` can include held-out effects with independent priors;
being listed there does not establish observed-target access. Explicit
recalibration can make a formerly eligible member ineligible. Nominal query
partitions alone never establish eligibility after additional target access.

Fitted models record their access automatically. A manually constructed
model with empty `training_pairs={}` has unknown encoder access and is
excluded conservatively. A genuinely untrained encoder can explicitly declare
`training_pairs={region.name: ()}`. This is a declaration of no target access,
not a way to erase a fitted model's history. Validation and calibration access
must likewise remain truthful.

## Known effects and support conditioning

Known effects require an explicit mode:

```python
from ilg_toolkit import EvaluationRegime

regime = EvaluationRegime(
    endpoint_regime="at_least_one_unseen", prediction_mode="known_effects"
)
prediction = predict_out_of_fold(ensemble, region, query_pairs, regime=regime)
```

Stored effects retain their actual calibration target access. A held-out
population with a prior-only effect can remain eligible; known-effect mode does
not silently relax endpoint or own-target rules.

Support prediction requires exactly declared support observations and a
support-role partition, disjoint from all requested query pairs:

```python
from ilg_toolkit import EvaluationSupport

support = EvaluationSupport(support_observations, support_partition)
regime = EvaluationRegime(
    prediction_mode="support", support_endpoint_policy="allow_declared_support"
)
prediction = predict_out_of_fold(
    ensemble, region, query_pairs, regime=regime, support=support
)
```

For several regions, map support declarations by region. Support observations
must contain exactly the support partition. Supplying support without support
mode, undeclared support, mismatched target metadata or support/query overlap
fails explicitly. Each member also rejects reusing its calibration targets as
support. Actual conditional provenance must match the declared mode and target
access before its values can contribute.

The default support endpoint policy, `unseen`, counts support endpoints as seen,
so strict untouched queries may lose coverage. `allow_declared_support` explicitly
reports conditional prediction: endpoint holdout is established from training,
selection and calibration **before** disjoint support is consumed. It does not
claim those endpoints remain untouched afterward. The own-query target remains
forbidden, and the returned regime/access/provenance retain this distinction.

`member_model_variances` retains conditional per-member variance separately on
the transformed model scale. It includes effect posterior, residual and configured
jitter variance, while excluding encoder, fixed-coefficient and variance-parameter
uncertainty. Member disagreement is descriptive in original units; these outputs
are not combined into a confidence interval or silently inverse-transformed into
original-scale Gaussian uncertainty.
