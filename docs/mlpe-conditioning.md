# Maximum-likelihood population effects (MLPE) prediction

A landscape genetics model with an `MLPECalibrationModule` uses the fitted
calibration for the query region. `model.predict(region)` gives marginal genetic
predictions, with population-effect means equal to zero.
Genetic calibration for a frozen distance model is fitted with `fit`, passing the
model as `distance_model` and `TrainingConfig(objective="mlpe", steps=0)` as `config`.

`model.predict(region, pairs=pairs, known_effects=True)` uses the saved population
effects. Unseen sampling units receive zero-mean population effects with the
fitted population variance. The query region supplies locations for every
sampling unit and the same feature meanings and order used for training.

Support observations update the saved population effects while holding the
distance model, calibration coefficients and variance parameters fixed:

```python
from ilg_toolkit import PairwiseObservations

support = PairwiseObservations.from_pairs(
    [("new-north", "known-west"), ("new-north", "known-east")],
    [0.8, 1.1],
    target=model.target,
)
prediction = model.predict(
    region, pairs=[("new-north", "known-south")], support=support,
)
```

`known_effects=True` and `support=` are mutually exclusive and require an MLPE
calibration for the region. Support targets must match the calibration's target.
Support pairs must be disjoint from calibration and query pairs, including
reversed pairs. Omitted `pairs` requests every distinct pair, so support
conditioning requires a disjoint query selection. Sampling-unit identifiers
define observation identities; callers resolve aliases before preparing inputs.

Both conditional modes return a `Prediction` on the original scale with three
per-pair arrays in `extras`:

| Name | Meaning |
| --- | --- |
| `model_values` | Conditional genetic predictions on the model scale |
| `effect_variance` | Variance from the conditioned population effects |
| `model_variance` | Effect variance plus residual variance and configured jitter |

Residual variance is available as
`model.calibration.regions[region.name].residual_variance`, and jitter as
`model.calibration.regions[region.name].config.jitter`. The variance calculations
treat the distance model, coefficients and variance parameters as fixed.
Conditional calculations use float64 within the call; resistance calculations
require float64 to be enabled before prediction.

Values are inverse-transformed model means. With a nonlinear target transform,
they are not expectations on the original scale; variance stays on the model
scale. `prediction.pairs` retains requested order and orientation, and
`prediction.as_matrix()` uses the region's sampling-unit order, with NaN on the
diagonal and for unrequested pairs. `prediction.as_matrix("effect_variance")`
rearranges the selected extra in the same way.

Calibration pairs and roles remain on the model's MLPE calibration. Support use
is recorded by [out-of-fold evaluation](evaluation.md) when assessing whether a
sampling unit's observations have been used. Conditional prediction alone
applies no held-out sampling-unit restriction.
