# Training losses and genetic prediction

Write $y_{ij}$ for an observed target in its declared units and
$t_{ij}=T(y_{ij})$ for its explicitly selected `TargetSpec` transform. Identity,
log1p and square root have different meanings; no conversion is inferred.
Ecological distances $s_{ij}$ are squared embedding distances or effective
resistances, depending on the distance model. They are distinct from genetic
predictions.

Observations can be supplied as labelled pairs or as a symmetric matrix whose
missing off-diagonal entries are `NaN`. Explicit training, validation,
calibration, support and query partitions record which measurements were
accessed. Validation determines the selected model, while optimization follows the fixed
requested step budget.

## Direct regression

Direct training compares the encoder output to the declared transformed target:

$$L_r=\frac{1}{m_r}\sum_{(i,j)\in\mathcal O_r}
 [\log(1+s_{ij})-\log(1+t_{ij})]^2.$$

Both inputs must be nonnegative dissimilarities. The log1p comparison is a loss
choice, independent of $T$; it emphasizes relative changes as values grow. Direct
prediction returns $T^{-1}(s)$ without fitting an MLPE head. For multiple regions,
the objective is the equal-weight mean of $L_r$, so pair-rich regions do not
automatically dominate. Missing observations never contribute.

## Population MLPE

For each region, training distances are standardized using their observed-pair mean
$c$ and sample SD $a$ (ddof=1): $x_{ij}=(s_{ij}-c)/a$. MLPE assumes

$$t_{ij}=\alpha+\beta x_{ij}+b_i+b_j+\epsilon_{ij},\qquad
 b_i\overset{\mathrm{iid}}\sim N(0,u),\quad
 \epsilon_{ij}\overset{\mathrm{iid}}\sim N(0,d),\quad d=e+j.$$

Intercept $\alpha$ and slope $\beta$ are signed. Population variance $u$ and
residual variance $e$ are positive with explicit floors; $j$ is the declared
jitter addition, used consistently throughout fitting and prediction. It is
never increased automatically. The same observation model applies to embedding
and resistance distances. Individual relatedness requires another observation model.

A pair has variance $2u+d$. Two different pairs sharing one endpoint have
covariance $u$; disjoint pairs have covariance zero. Thus shared populations
induce dependent observations. With the pair-to-population incidence matrix $Z$
(two ones per row), integrating out the Gaussian population effects gives

$$t\sim N(X\theta,V),\qquad
 X=[\mathbf1,x],\quad\theta=(\alpha,\beta)^\top,\quad
 V=dI+uZZ^\top.$$

At fixed variances, conditional generalized least squares (GLS) profiles

$$\hat\theta=(X^\top V^{-1}X)^{-1}X^\top V^{-1}t.$$

The implementation uses linear solves rather than explicit inverses. The full
Gaussian negative log likelihood is

$$\ell=\tfrac12\{m\log(2\pi)+\log\det V+
 (t-X\hat\theta)^\top V^{-1}(t-X\hat\theta)\}.$$

This is maximum likelihood, without a REML correction. Standalone calibration
uses deterministic bounded multistart optimization to convergence. Its float64
JAX kernels provide exact derivatives to a host optimizer. Joint training instead
updates encoder and raw softplus variance parameters together with a fixed Adam
budget, minimizing the equal-region mean of $\ell_r/m_r$. It does not converge an
inner variance fit at each encoder update, and records `converged=False`.

After a joint update, training observations refresh $c,a,\hat\theta$ and the
population-effect posterior. Validation uses those frozen training quantities:
it does not recompute distance moments or refit coefficients on validation targets.
Constant distances, singular GLS, disjoint endpoint sets that cannot identify two
variance components, unsupported conditioning and failed optimization are explicit
failures. [Endpoint-system arithmetic and its precision limits](performance/mlpe.md)
explain the exact population-sized calculation.

## Fitted models and calibration modules

`fit` returns a `TrainingState` whose selected `.model` is a
`LandscapeGeneticsModel`. It composes `.distance_model` with `.calibration`:
`DirectCalibrationModule` applies the declared inverse target transform, while
`MLPECalibrationModule.regions` contains one `RegionalMLPECalibration` per fitted
region. The complete model's `.target` comes from its calibration module.

`fit(region, observations, distance_model=frozen, config=TrainingConfig(objective="mlpe", steps=0))`
fits calibration at step zero with the distance model fixed. Coefficients and
population effects use the training pairs and the initial variances in
`TrainingConfig.mlpe`, an `MLPEConfig`. The returned model covers the supplied
regions. Validation observations score that fitted calibration and select among
evaluated models.

## Prediction information

Query regions need only features and sampling-unit locations, with the training
feature meanings and order; locations are integer `(row, column)` cell positions.
`model.predict(region)` returns a `Prediction` with one original-scale value per
distinct pair, in region sampling-unit order. `predict(region, pairs=[("south", "east")])`
preserves the requested pair order and orientation. `prediction.as_matrix()`
rearranges stored values into a symmetric matrix in query-region sampling-unit
order. The diagonal and unrequested pairs are `NaN` for every target kind.
`as_matrix(name="effect_variance")` rearranges a named per-pair extra in the same
way when that extra is present. Target meaning and units remain on the model's
calibration.

Marginal prediction uses $\hat\alpha+\hat\beta x$ and zero population-effect
means, including for unseen populations. Known-effect prediction adds the saved
posterior means of the endpoints. Support-conditioned prediction first updates
that posterior from explicitly declared support observations while holding the
encoder, moments, fixed coefficients and variances fixed. Calibration pairs
cannot be counted twice, and support/query pairs must be disjoint.

For a fitted residual $r=t-X\hat\theta$, the saved Gaussian posterior is

$$E[b\mid t]=uZ^\top V^{-1}r,\qquad
 \operatorname{Cov}(b\mid t)=(u^{-1}I+d^{-1}Z^\top Z)^{-1}.$$

Known/support prediction includes endpoint posterior variance and independent
new-observation residual variance $d$. It excludes encoder, fixed-coefficient and
variance-parameter uncertainty. Unseen effects have independent prior variance
$u$. Numerical means and variances are JAX arrays; identity/provenance remains
host metadata. See [support conditioning](mlpe-conditioning.md) for access rules.

All point predictions return $T^{-1}$ of the fitted mean in original units,
without clipping. Under nonlinear $T$, this is not the expectation of the
original random target. Conditional variance remains on the fitted model scale;
ensemble member spread is a different descriptive quantity.

The separate [experimental Wishart contract](statistics/wishart.md) records its
likelihood assumptions and scoring derivation; its diagnostic does not enable
Wishart training.
