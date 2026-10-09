"""Pure differentiable full-ML likelihood; labels and optimization live elsewhere.

Input array dtypes control precision. Use ``jax.enable_x64()`` and float64 arrays
for reference comparisons; importing this module never changes global precision.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

DEFAULT_VARIANCE_FLOOR = 1e-10
DEFAULT_MIN_SCORE_SCALE = 1e-12


def pair_incidence_matrix(left, right, *, n_populations, dtype=jnp.float32):
    """Two endpoint effects per observation, in the caller's population order."""
    rows = jnp.arange(len(left))
    incidence = jnp.zeros((len(left), n_populations), dtype=dtype)
    return incidence.at[rows, left].add(1).at[rows, right].add(1)


def sample_standardize_scores(scores, *, center=None, scale=None):
    """R-compatible sample SD (ddof=1), optionally using stored training moments."""
    scores = jnp.asarray(scores)
    center = jnp.mean(scores) if center is None else jnp.asarray(center, dtype=scores.dtype)
    scale = (
        jnp.sqrt(jnp.sum((scores - center) ** 2) / (len(scores) - 1))
        if scale is None
        else jnp.asarray(scale, dtype=scores.dtype)
    )
    return (scores - center) / scale, center, scale


def decode_mlpe_variances(raw_variances, *, variance_floor=DEFAULT_VARIANCE_FLOOR):
    """Decode raw softplus parameters as (unit variance, residual variance)."""
    positive = jax.nn.softplus(jnp.asarray(raw_variances)) + variance_floor
    return positive[0], positive[1]


def _components(
    scores,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    score_center,
    score_scale,
    variance_floor,
    jitter,
):
    scores = jnp.asarray(scores)
    standardized, _, scale = sample_standardize_scores(
        scores, center=score_center, scale=score_scale
    )
    design = jnp.stack([jnp.ones_like(scores), standardized], axis=1)
    z = pair_incidence_matrix(left, right, n_populations=n_populations, dtype=scores.dtype)
    unit, residual = decode_mlpe_variances(raw_variances, variance_floor=variance_floor)
    covariance = unit * (z @ z.T) + (residual + jitter) * jnp.eye(len(scores), dtype=scores.dtype)
    return jnp.asarray(targets, dtype=scores.dtype), design, covariance, scale


def _nll(targets, design, factor, beta):
    residual = targets - design @ beta
    return 0.5 * (
        len(targets) * math.log(2 * math.pi)
        + 2 * jnp.log(jnp.diag(factor)).sum()
        + residual @ cho_solve((factor, True), residual)
    )


def profiled_mlpe_ml_fit(
    scores,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    score_center=None,
    score_scale=None,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
):
    """Return full Gaussian ML NLL and signed GLS coefficients.

    Fixed effects are profiled at the supplied variances. Constant/invalid scores
    or singular GLS return nonfinite results for the host boundary to report.
    This is ML rather than REML. NLL is not normalized by observation count.
    """
    targets, design, covariance, scale = _components(
        scores,
        targets,
        left,
        right,
        n_populations=n_populations,
        raw_variances=raw_variances,
        score_center=score_center,
        score_scale=score_scale,
        variance_floor=variance_floor,
        jitter=jitter,
    )
    factor = jnp.linalg.cholesky(covariance)
    vinv_design = cho_solve((factor, True), design)
    beta = jnp.linalg.solve(design.T @ vinv_design, design.T @ cho_solve((factor, True), targets))
    # Envelope theorem: the conditional optimum has zero beta derivative.
    nll = _nll(targets, design, factor, jax.lax.stop_gradient(beta))
    valid = jnp.isfinite(scale) & (scale > DEFAULT_MIN_SCORE_SCALE) & jnp.isfinite(beta).all()
    return jnp.where(valid, nll, jnp.nan), jnp.where(valid, beta, jnp.nan)


def profiled_mlpe_ml_negative_log_likelihood(*args, **kwargs):
    """Full-ML NLL after profiling the signed intercept and slope."""
    return profiled_mlpe_ml_fit(*args, **kwargs)[0]


def mlpe_ml_negative_log_likelihood(
    scores,
    targets,
    left,
    right,
    *,
    n_populations,
    fixed_effects,
    raw_variances,
    score_center=None,
    score_scale=None,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
):
    """Full ML with frozen fixed effects and score moments, for validation."""
    targets, design, covariance, scale = _components(
        scores,
        targets,
        left,
        right,
        n_populations=n_populations,
        raw_variances=raw_variances,
        score_center=score_center,
        score_scale=score_scale,
        variance_floor=variance_floor,
        jitter=jitter,
    )
    nll = _nll(targets, design, jnp.linalg.cholesky(covariance), jnp.asarray(fixed_effects))
    return jnp.where(jnp.isfinite(scale) & (scale > DEFAULT_MIN_SCORE_SCALE), nll, jnp.nan)
