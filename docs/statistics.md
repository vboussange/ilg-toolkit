# Training losses and genetic prediction

Let $y_{ij}$ be an observed genetic dissimilarity and $t_{ij}=T(y_{ij})$ its value
under the [target transformation](api/data.md#ilg_toolkit.TargetSpec).
Ecological distances $s_{ij}$ come from effective resistance or squared embedding
distance. Choose a loss explicitly with `TrainingConfig(loss=...)`.

## Direct log1p regression

[`Log1pLoss`](api/training.md#ilg_toolkit.Log1pLoss) compares nonnegative ecological
distances and transformed targets:

$$L_r=\frac{1}{m_r}\sum_{(i,j)\in\mathcal O_r}
[\log(1+s_{ij})-\log(1+t_{ij})]^2.$$

The log1p comparison and target transform are separate choices. Direct calibration
returns $T^{-1}(s)$ and has no fitted coefficients or variance parameters. Training
averages regional losses equally within each species, then averages species equally.

## Maximum-likelihood population effects

[`MLPELoss`](api/training.md#ilg_toolkit.MLPELoss) accounts for dependence between
measurements sharing a population. With sample-standardized ecological distances
$x_{ij}=(s_{ij}-c)/a$,

$$t_{ij}=\alpha+\beta x_{ij}+b_i+b_j+\epsilon_{ij},\qquad
b_i\sim N(0,u),\quad\epsilon_{ij}\sim N(0,e+j).$$

The intercept and slope are **signed**; $u$ and $e$ are positive population and
residual variances, and $j$ is fixed numerical jitter. A pair has variance
$2u+e+j$; distinct pairs sharing a population have covariance $u$. Coefficients,
variance components, distance moments, and population-effect posteriors are
estimated separately for each region–species group. Distance-model parameters
can be pooled across those groups.

With incidence matrix $Z$ and $X=[\mathbf1,x]$, the covariance is
$V=(e+j)I+uZZ^\top$. Generalized least squares profiles the coefficients, and
training minimizes the marginal negative log likelihood per observed pair:

$$\ell=\tfrac12\{m\log(2\pi)+\log\det V+
(t-X\hat\theta)^\top V^{-1}(t-X\hat\theta)\}.$$

This is maximum likelihood, without restricted-likelihood correction. `train`
updates distance and variance parameters with a fixed Adam budget. Even at zero
steps it is **initialization**, not converged variance fitting. `calibrate` keeps
the distance model fixed and optimizes both variance components using deterministic
multistart optimization. See [advanced MLPE](api/mlpe.md) for identifiability,
precision, diagnostics, and the standalone array-distance fitter.

## Evaluation answers a specified question

Built-in validation uses the training-fitted calibration to select an evaluated
training step. For explanatory association on unseen populations, the
[MLPE workflow](examples/mlpe_workflow.md) implements a different design:
train distance parameters on training populations; fit fresh validation calibration
for each frozen candidate; select by marginal-log-likelihood improvement over a
geographic MLPE baseline; then freeze the selected distance parameters and fit
fresh test calibration. Both baseline and landscape fits use matching targets,
observations, settings, and optimizer budgets.

Test observations fit and score the test calibration. The score describes
association of a representation learned without those populations' genetic
observations; it is not untouched end-to-end predictive accuracy. Inspect signed
slopes, both variances, convergence, and population/pair counts across repeated
splits. A favorable likelihood alone does not establish the expected direction.
`IBDModel` fits a separate log1p geographic regression and is not the matching MLPE
likelihood baseline. A constant-distance intercept-only null is unsupported because
the MLPE distance design must be identifiable.

## Prediction and conditioning

The [prediction API](api/prediction.md) gives worked examples and constraints.
Marginal predictions use zero-mean population effects. `known_effects=True` uses
the saved posterior; `support=...` updates that posterior with additional,
disjoint observations while keeping the distance model, coefficients, moments,
and variances fixed. A new region–species relationship requires fresh calibration.

Genetic predictions apply the inverse target transform. With nonlinear transforms,
this transformed fitted mean is not automatically the expectation on the original
scale. Population-effect uncertainty describes that posterior conditional on the
fitted model; ensemble spread describes variability among members. Neither alone
quantifies all model or ecological uncertainty.

## Wishart diagnostic (experimental)

The research-only [Wishart experiment](https://github.com/vboussange/ilg-toolkit/tree/main/experiments/wishart)
scores a **fixed-parameter** Gaussian marker-contrast model. It assumes independent,
zero-mean Gaussian marker contrasts and a complete squared-distance matrix with
explicit scatter-versus-average scaling and a known marker count. The ordinary
Wishart diagnostic requires a positive-definite contrast scatter; missing pairs,
insufficient marker rank, or unmodeled marker dependence do not satisfy it.

This experiment is not a toolkit loss or calibration method and performs no
Wishart fitting. It provides a reproducible diagnostic reference for comparing
fixed covariance hypotheses under those assumptions.
