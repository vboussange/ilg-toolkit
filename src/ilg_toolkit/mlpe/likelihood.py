"""Exact population-sized full-ML likelihood and signed profiled GLS.

Input array dtypes control precision. Use ``jax.enable_x64()`` and float64 arrays
for reference comparisons; importing this module never changes global precision.
Invalid/singular inputs or unsupported numerical conditioning return nonfinite
results for the host boundary to report. No diagonal is increased automatically.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from .system import population_system

DEFAULT_VARIANCE_FLOOR = 1e-10
DEFAULT_MIN_SCORE_SCALE = 1e-12


def pair_incidence_matrix(left, right, *, n_populations, dtype=jnp.float32):
    """Explicit incidence utility; ordinary fitting and prediction use scatters."""
    rows = jnp.arange(len(left))
    incidence = jnp.zeros((len(left), n_populations), dtype=dtype)
    return incidence.at[rows, left].add(1).at[rows, right].add(1)


def sample_standardize_scores(scores, *, center=None, scale=None, pair_mask=None):
    """Sample SD (ddof=1) of observed scores, or stored training moments."""
    scores = jnp.asarray(scores)
    weights = jnp.ones(scores.shape, dtype=bool) if pair_mask is None else jnp.asarray(pair_mask)
    safe_scores = jnp.where(weights, scores, 0)
    count = jnp.sum(weights)
    center = (
        jnp.sum(safe_scores) / count if center is None else jnp.asarray(center, dtype=scores.dtype)
    )
    differences = jnp.where(weights, safe_scores - center, 0)
    scale = (
        jnp.sqrt(jnp.sum(differences**2) / (count - 1))
        if scale is None
        else jnp.asarray(scale, dtype=scores.dtype)
    )
    return differences / scale, center, scale


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
    pair_mask,
):
    dtype = jnp.result_type(scores, targets, raw_variances, jnp.float32)
    scores = jnp.asarray(scores, dtype=dtype)
    targets = jnp.asarray(targets, dtype=dtype)
    left, right = jnp.asarray(left, dtype=jnp.int32), jnp.asarray(right, dtype=jnp.int32)
    if (
        scores.ndim != 1
        or targets.shape != scores.shape
        or left.shape != scores.shape
        or right.shape != scores.shape
        or n_populations < 2
    ):
        raise ValueError("MLPE requires aligned one-dimensional scores, targets and endpoints")
    weights = jnp.ones(scores.shape, dtype=bool) if pair_mask is None else jnp.asarray(pair_mask)
    if weights.shape != scores.shape or weights.dtype != jnp.bool_:
        raise ValueError("pair_mask must be a boolean vector aligned with scores")
    standardized, _, scale = sample_standardize_scores(
        scores,
        center=score_center,
        scale=score_scale,
        pair_mask=weights,
    )
    design = jnp.stack([weights.astype(scores.dtype), standardized], axis=1)
    unit, residual = decode_mlpe_variances(raw_variances, variance_floor=variance_floor)
    system = population_system(
        left,
        right,
        n_populations=n_populations,
        unit=unit,
        residual=residual + jitter,
        weights=weights,
        dtype=scores.dtype,
    )
    targets = jnp.where(weights, targets, 0)
    valid = (
        system.valid
        & jnp.isfinite(scale)
        & (scale > DEFAULT_MIN_SCORE_SCALE)
        & jnp.isfinite(design).all()
        & jnp.isfinite(targets).all()
        & (system.pair_count >= 1)
    )
    return targets, design, system, valid


def _nll(system, remaining, effects):
    quadratic = jnp.sum(remaining**2) / system.residual + jnp.sum(effects**2) / system.unit
    return 0.5 * (system.pair_count * math.log(2 * math.pi) + system.logdet + quadratic)


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
    pair_mask=None,
):
    """Return unnormalized full ML NLL and signed GLS coefficients.

    ``pair_mask`` is JIT-compatible: false rows, including NaN measurements and
    placeholder endpoints, are absent from moments, determinants and quadratics.
    Conditioning beyond eps*(1+2*max_degree*u/(e+j)) <= .01 is rejected with NaN;
    use float64 or explicit scientifically appropriate variance constraints.
    """
    targets, design, system, valid = _components(
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
        pair_mask=pair_mask,
    )
    rhs = jnp.column_stack((design, targets))
    inner, remaining, effects = system.inner_products(rhs)
    beta = jnp.linalg.solve(inner[:2, :2], inner[:2, 2])
    beta_for_likelihood = jax.lax.stop_gradient(beta)
    residual_remaining = remaining[:, 2] - remaining[:, :2] @ beta_for_likelihood
    residual_effects = effects[:, 2] - effects[:, :2] @ beta_for_likelihood
    nll = _nll(system, residual_remaining, residual_effects)
    valid = (
        valid
        & system.identifiable
        & (system.pair_count >= 3)
        & jnp.isfinite(beta).all()
        & jnp.isfinite(nll)
    )
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
    pair_mask=None,
):
    """Full ML with frozen fixed effects and score moments, for validation."""
    targets, design, system, valid = _components(
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
        pair_mask=pair_mask,
    )
    residual = targets - design @ jnp.asarray(fixed_effects)
    remaining, effects = system.residual_components(residual[:, None])
    nll = _nll(system, remaining, effects)
    return jnp.where(valid & jnp.isfinite(nll), nll, jnp.nan)


def mlpe_effect_posterior(
    scores,
    targets,
    left,
    right,
    *,
    n_populations,
    fixed_effects,
    raw_variances,
    score_center,
    score_scale,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
    pair_mask=None,
):
    """Return effect mean, covariance, and precision factor at current parameters.

    This refresh operation does not optimize variances or refit the encoder.
    It uses the same population system and numerical policy as the likelihood.
    """
    targets, design, system, valid = _components(
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
        pair_mask=pair_mask,
    )
    residual = targets - design @ jnp.asarray(fixed_effects)
    _, effects = system.residual_components(residual[:, None])
    covariance = system.unit * cho_solve(
        (system.factor, True), jnp.eye(n_populations, dtype=design.dtype)
    )
    factor = system.factor / jnp.sqrt(system.unit)
    return (
        jnp.where(valid, effects[:, 0], jnp.nan),
        jnp.where(valid, (covariance + covariance.T) / 2, jnp.nan),
        jnp.where(valid, factor, jnp.nan),
    )
