# Training losses and genetic prediction

Write $y_{ij}$ for an observed target in its declared units and
$t_{ij}=T(y_{ij})$ for its explicitly selected `TargetSpec` transform. Identity,
log1p and square root have different meanings; no conversion is inferred.
Landscape scores $s_{ij}$ are squared embedding distances or effective
resistances, depending on the encoder. They are distinct from genetic predictions.

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

For each region, training scores are standardized using their observed-pair mean
$c$ and sample SD $a$ (ddof=1): $x_{ij}=(s_{ij}-c)/a$. MLPE assumes

$$t_{ij}=\alpha+\beta x_{ij}+b_i+b_j+\epsilon_{ij},\qquad
 b_i\overset{\mathrm{iid}}\sim N(0,u),\quad
 \epsilon_{ij}\overset{\mathrm{iid}}\sim N(0,d),\quad d=e+j.$$

Intercept $\alpha$ and slope $\beta$ are signed. Population variance $u$ and
residual variance $e$ are positive with explicit floors; $j$ is the declared
jitter addition, used consistently throughout fitting and prediction. It is
never increased automatically. The same observation model applies to embedding
and resistance scores. Individual relatedness requires another observation model.

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
it does not recompute score moments or refit coefficients on validation targets.
Constant scores, singular GLS, disjoint endpoint sets that cannot identify two
variance components, unsupported conditioning and failed optimization are explicit
failures. [Endpoint-system arithmetic and its precision limits](performance/mlpe.md)
explain the exact population-sized calculation.

## Prediction information

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
