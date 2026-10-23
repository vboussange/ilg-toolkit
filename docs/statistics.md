# Training losses and genetic prediction

Let $y_{ij}$ be a pairwise genetic observation on its original scale and
$t_{ij}=T(y_{ij})$ its model-scale value under the `TargetSpec` transform.
Ecological distances $s_{ij}$ are effective resistance or squared embedding
distance. `TrainingConfig.objective` selects direct regression or
maximum-likelihood population effects (MLPE).

Training observations determine the fitted distance model and genetic calibration.
Validation observations select among evaluated training steps. Query observations
score predictions afterward. [Evaluation](evaluation.md) explains these data roles
for fold ensembles.

## Direct regression

For $m_r$ observed pairs in region $r$, direct training minimizes

$$L_r=\frac{1}{m_r}\sum_{(i,j)\in\mathcal O_r}
 [\log(1+s_{ij})-\log(1+t_{ij})]^2.$$

Both inputs are nonnegative dissimilarities. The log1p comparison emphasizes
relative changes as values grow; the target transform $T$ is configured
separately. Direct calibration predicts $T^{-1}(s)$.

For one species, training averages regional losses equally. Experimental
multispecies training gives species equal weight and averages regional losses
within each species. Missing observations are excluded from these losses.

## Population MLPE

Within each region, observed-pair distances have mean $c$ and sample standard
deviation $a$ (`ddof=1`). Standardized distances are $x_{ij}=(s_{ij}-c)/a$, and

$$t_{ij}=\alpha+\beta x_{ij}+b_i+b_j+\epsilon_{ij},\qquad
 b_i\overset{\mathrm{iid}}\sim N(0,u),\quad
 \epsilon_{ij}\overset{\mathrm{iid}}\sim N(0,d),\quad d=e+j.$$

Population effects $b_i$ and residuals $\epsilon_{ij}$ are independently and
identically distributed within their respective groups. Here $u$ is population
variance, $e$ is residual variance, and $j$ is configured jitter. Intercept $\alpha$ and slope $\beta$ are
signed. `MLPEConfig` defines positive variance floors and jitter. The observation
model supports population dissimilarities with either distance-model family.

A pair has variance $2u+d$. Distinct pairs sharing a population have covariance
$u$; disjoint pairs have covariance zero. With pair-to-population incidence matrix
$Z$ containing two ones per row,

$$t\sim N(X\theta,V),\qquad
 X=[\mathbf1,x],\quad\theta=(\alpha,\beta)^\top,\quad V=dI+uZZ^\top.$$

At fixed variances, generalized least squares (GLS) gives

$$\hat\theta=(X^\top V^{-1}X)^{-1}X^\top V^{-1}t,$$

and the Gaussian negative log likelihood is

$$\ell=\tfrac12\{m\log(2\pi)+\log\det V+
 (t-X\hat\theta)^\top V^{-1}(t-X\hat\theta)\}.$$

Joint training updates distance-model parameters and variance parameters with
Adam. It minimizes the equal-region mean of $\ell_r/m_r$ for one species, or the
equal-species mean of within-species regional means for multispecies fits. Each
region–species group has unpooled genetic calibration. At evaluations, training
observations determine $c,a,\hat\theta$ and the population-effect posterior;
validation uses those fitted quantities.

`fit(..., config=TrainingConfig(objective="mlpe", steps=0))` keeps the distance
model fixed and fits coefficients and population effects using the initial
variances in `TrainingConfig.mlpe`. It returns calibration for the supplied
regions. The standalone `calibrate_mlpe(distances, observations, ...)` also
optimizes variance parameters to convergence. Joint and zero-step fits report
`converged=False`; their budget is determined by `steps`.

Note: This is maximum likelihood, without a restricted-likelihood correction.
Individual relatedness requires a different observation model.

Constant distances, singular GLS systems, and observations that cannot identify
the variance components are rejected. [Performance](performance.md#mlpe-calculations)
describes the population-sized calculation and precision limits.

## MLPE conditioning

`model.predict(region)` uses marginal genetic predictions with model-scale mean
$\hat\alpha+\hat\beta x$. `model.predict(region, known_effects=True)` adds saved
population effects. For fitted residual $r=t-X\hat\theta$, their posterior is

$$E[b\mid t]=uZ^\top V^{-1}r,\qquad
 \operatorname{Cov}(b\mid t)=(u^{-1}I+d^{-1}Z^\top Z)^{-1}.$$

Unseen sampling units receive zero-mean effects with prior variance $u$.
Support observations condition these effects while keeping the distance model,
distance moments, coefficients, and variance parameters fixed:

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

Both modes require MLPE calibration for the query region; `known_effects=True`
and `support=` are mutually exclusive. Support targets must match `model.target`.
Support pairs must be disjoint from calibration and query pairs, including
reversed pairs. Since omitted `pairs` requests every distinct pair, support
conditioning requires a query selection disjoint from support. The query region
provides locations for all sampling units and the training feature meanings and
order. Identifiers determine observation identity; resolve aliases when preparing
data.

Conditional predictions return a `Prediction` with pair-aligned `extras`:

| Name | Meaning |
| --- | --- |
| `model_values` | Conditional mean on the model scale |
| `effect_variance` | Variance from conditioned population effects |
| `model_variance` | Effect variance plus residual variance and jitter |

For a single-species `MLPECalibrationModule`,
`model.calibration.regions[region.name]` provides `residual_variance`, `config.jitter`,
and calibration observation-use records. For experimental multispecies models,
`model.calibration.for_region(region.name, species_index)` selects the calibration
using the index in `model.species`.

Conditional calculations use float64 within the call. Resistance calculations
and joint MLPE fitting require float64 enabled before constructing inputs.
`prediction.as_matrix("effect_variance")` arranges a named extra in sampling-unit
order, with `NaN` on the diagonal and for unrequested pairs.

All genetic predictions apply $T^{-1}$ to the fitted mean and return the original
scale. Conditional variances remain on the model scale and treat distance-model
parameters, coefficients, and variance parameters as fixed. Ensemble member
spread describes disagreement on the original scale; see [evaluation](evaluation.md).

Note: For nonlinear $T$, inverse-transformed means are not expectations of the
original random target. Conditional prediction alone imposes no holdout restriction;
[out-of-fold evaluation](evaluation.md#holdouts-and-observation-use) checks observation use.

The [Wishart reference](statistics/wishart.md) describes experimental
fixed-parameter likelihood scoring.
