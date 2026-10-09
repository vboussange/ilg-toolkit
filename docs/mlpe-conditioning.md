# MLPE prediction and declared support

Ordinary calibrated prediction is marginal: population effects have mean zero.
It uses the saved regional intercept, signed slope and training score moments.
`head.predict_known_effects(scores, pairs)` is a separate operation that uses the
saved posterior effects for known populations. Unseen populations receive
independent zero-mean effects with the fitted population variance. Scores align
with the supplied labelled pairs; no query genetic observations are accepted.

Support observations explicitly update those population effects while keeping
the encoder, coefficients and variance parameters fixed:

```python
from ilg_toolkit import ObservationPartition, PairwiseObservations

# head is an existing regional MLPE calibration. Scores come from its frozen encoder.
support = PairwiseObservations.from_pairs(
    [("new-north", "known-west"), ("new-north", "known-east")],
    [0.8, 1.1],
    target=head.target,
)
declared = ObservationPartition(head.region_name, support.observed_pairs, role="support")
# support_scores must follow support.observed_pairs, which can reorder input rows.
conditioned = head.condition_on_support(support_scores, support, partition=declared)
prediction = conditioned.predict(query_scores, [("new-north", "known-south")])
```

The standalone functions `predict_known_effects(head, scores, pairs)` and
`condition_on_support(head, support_scores, support_observations, partition=...)`
expose the same operations. Support observations must contain exactly the declared
partition: extra genetic targets are refused. The region and target metadata must
match the calibration. Missing observations remain absent.

Unordered endpoint labels are observation identities. Reversing endpoints or
reordering input rows cannot bypass the guards. Reusing a calibration pair as
support would count the same genetic target twice and is refused. A query cannot
reuse a support pair, and duplicate queries, including reversed duplicates, are
refused. New distinct labels denote distinct populations; callers must resolve
aliases for the same physical population before preparing the inputs.

The support update is exact Gaussian conditioning on the saved effect posterior.
It factors a population-sized whitened precision system, without forming a
support-pair covariance matrix. The effective residual variance is the fitted
residual variance plus explicitly configured jitter throughout calibration,
support updating and prediction. No numerical jitter is increased automatically.

`MLPEConditionalPrediction.model_variance` describes a new independent genetic
observation: it includes population-effect posterior uncertainty, residual
variance and configured jitter. `effect_variance`, `residual_variance` and
`jitter` expose these components. The encoder, fixed coefficients and variance
parameters are treated as fixed; their uncertainty is explicitly excluded.
The variance scale is `model`, meaning the target scale after its declared
transform (squared transformed units, or squared original units for identity).

Point predictions in `values` are `target.inverse(model_values)` on original
measurement units. For nonlinear transforms they are inverse-transformed Gaussian
means, rather than expectations on the original scale. The reported variance
remains on the model scale. The API does not construct an original-scale Gaussian
confidence interval, and predictions are neither clipped nor repaired.

Prediction provenance records the region, prediction mode, canonical calibration
pairs and their roles, supplied support pairs and their support roles, and the
independent-prior unseen-effect policy. These identities complement the predictor's
encoder-training and validation provenance when deciding evaluation eligibility.
Known-effect prediction may predict a calibration pair for deployment; provenance
must still exclude that target from a claim of held-out evaluation. Statistical
predictive variance and ensemble disagreement describe different uncertainty.
