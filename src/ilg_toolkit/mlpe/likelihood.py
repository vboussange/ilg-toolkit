"""Maximum-likelihood population effects (MLPE) likelihood and calibration.

Generalized least squares (GLS) fits the calibration coefficients.

Input array data types control precision; use ``jax.enable_x64()`` and float64
arrays for reference comparisons.
Invalid inputs, singular systems, or unsupported numerical conditioning return
nonfinite results.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from .covariance import population_covariance_factor

DEFAULT_VARIANCE_FLOOR = 1e-10
DEFAULT_MIN_DISTANCE_SCALE = 1e-12


class MLPEFit(NamedTuple):
    """MLPE calibration coefficients, likelihood, and population effects at supplied variances."""

    nll: jax.Array
    coefficients: jax.Array
    distance_center: jax.Array
    distance_scale: jax.Array
    variances: jax.Array
    posterior: tuple[jax.Array, jax.Array, jax.Array]


def pair_incidence_matrix(left, right, *, n_populations, dtype=jnp.float32):
    """Pair-to-population incidence matrix with two ones per observed pair."""
    rows = jnp.arange(len(left))
    incidence = jnp.zeros((len(left), n_populations), dtype=dtype)
    return incidence.at[rows, left].add(1).at[rows, right].add(1)


def sample_standardize_distances(distances, *, center=None, scale=None, pair_mask=None):
    """Standardized ecological distances with their mean and sample standard deviation."""
    distances = jnp.asarray(distances)
    weights = jnp.ones(distances.shape, dtype=bool) if pair_mask is None else jnp.asarray(pair_mask)
    safe_distances = jnp.where(weights, distances, 0)
    count = jnp.sum(weights)
    center = (
        jnp.sum(safe_distances) / count
        if center is None
        else jnp.asarray(center, dtype=distances.dtype)
    )
    differences = jnp.where(weights, safe_distances - center, 0)
    scale = (
        jnp.sqrt(jnp.sum(differences**2) / (count - 1))
        if scale is None
        else jnp.asarray(scale, dtype=distances.dtype)
    )
    return differences / scale, center, scale


def decode_mlpe_variances(raw_variances, *, variance_floor=DEFAULT_VARIANCE_FLOOR):
    """Population and residual variance parameters obtained through softplus."""
    positive = jax.nn.softplus(jnp.asarray(raw_variances)) + variance_floor
    return positive[0], positive[1]


def _components(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    distance_center,
    distance_scale,
    min_distance_scale,
    variance_floor,
    jitter,
    pair_mask,
):
    distances, targets, raw_variances = map(jnp.asarray, (distances, targets, raw_variances))
    dtype = jnp.result_type(distances, targets, raw_variances, jnp.float32)
    distances = jnp.asarray(distances, dtype=dtype)
    targets = jnp.asarray(targets, dtype=dtype)
    left, right = jnp.asarray(left, dtype=jnp.int32), jnp.asarray(right, dtype=jnp.int32)
    if (
        distances.ndim != 1
        or targets.shape != distances.shape
        or left.shape != distances.shape
        or right.shape != distances.shape
        or n_populations < 2
    ):
        raise ValueError(
            "MLPE requires aligned one-dimensional distances, targets and sampling units"
        )
    weights = jnp.ones(distances.shape, dtype=bool) if pair_mask is None else jnp.asarray(pair_mask)
    if weights.shape != distances.shape or weights.dtype != jnp.bool_:
        raise ValueError("pair_mask must be a boolean vector aligned with distances")
    standardized, center, scale = sample_standardize_distances(
        distances,
        center=distance_center,
        scale=distance_scale,
        pair_mask=weights,
    )
    design = jnp.stack([weights.astype(distances.dtype), standardized], axis=1)
    unit, residual = decode_mlpe_variances(raw_variances, variance_floor=variance_floor)
    system = population_covariance_factor(
        left,
        right,
        n_populations=n_populations,
        unit=unit,
        residual=residual + jitter,
        weights=weights,
        dtype=distances.dtype,
    )
    targets = jnp.where(weights, targets, 0)
    valid = (
        system.valid
        & jnp.isfinite(scale)
        & (scale > min_distance_scale)
        & jnp.isfinite(design).all()
        & jnp.isfinite(targets).all()
        & (system.pair_count >= 1)
    )
    return targets, design, system, valid, center, scale


def _nll(system, remaining, effects):
    quadratic = jnp.sum(remaining**2) / system.residual + jnp.sum(effects**2) / system.unit
    return 0.5 * (system.pair_count * math.log(2 * math.pi) + system.logdet + quadratic)


def _profile(targets, design, system, valid):
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
    return nll, beta, effects, valid


def _posterior(system, effects, valid):
    covariance = system.unit * cho_solve(
        (system.factor, True), jnp.eye(len(system.factor), dtype=system.factor.dtype)
    )
    factor = system.factor / jnp.sqrt(system.unit)
    return (
        jnp.where(valid, effects, jnp.nan),
        jnp.where(valid, (covariance + covariance.T) / 2, jnp.nan),
        jnp.where(valid, factor, jnp.nan),
    )


def profiled_mlpe_ml_fit(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    distance_center=None,
    distance_scale=None,
    min_distance_scale=DEFAULT_MIN_DISTANCE_SCALE,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
    pair_mask=None,
):
    """Unnormalized negative log likelihood and signed GLS coefficients for MLPE calibration.

    ``pair_mask`` selects observations for distance moments and likelihood calculations.
    Numerical conditioning must satisfy eps*(1+2*max_degree*u/(e+j)) <= 0.01;
    unsupported inputs return not-a-number (NaN) results.
    """
    targets, design, system, valid, _, _ = _components(
        distances,
        targets,
        left,
        right,
        n_populations=n_populations,
        raw_variances=raw_variances,
        distance_center=distance_center,
        distance_scale=distance_scale,
        min_distance_scale=min_distance_scale,
        variance_floor=variance_floor,
        jitter=jitter,
        pair_mask=pair_mask,
    )
    nll, beta, _, valid = _profile(targets, design, system, valid)
    return jnp.where(valid, nll, jnp.nan), jnp.where(valid, beta, jnp.nan)


def finalize_mlpe_fit(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    distance_center=None,
    distance_scale=None,
    min_distance_scale=DEFAULT_MIN_DISTANCE_SCALE,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
    pair_mask=None,
) -> MLPEFit:
    """MLPE coefficients, likelihood, and population effects at supplied variances.

    The population system is factored once and reused for all fitted quantities.
    Distance moments use selected observations or the supplied training moments.
    ``profiled_mlpe_ml_fit`` provides the likelihood and coefficients for gradient optimization.
    """
    targets, design, system, valid, center, scale = _components(
        distances,
        targets,
        left,
        right,
        n_populations=n_populations,
        raw_variances=raw_variances,
        distance_center=distance_center,
        distance_scale=distance_scale,
        min_distance_scale=min_distance_scale,
        variance_floor=variance_floor,
        jitter=jitter,
        pair_mask=pair_mask,
    )
    nll, beta, effects, valid = _profile(targets, design, system, valid)
    return MLPEFit(
        jnp.where(valid, nll, jnp.nan),
        jnp.where(valid, beta, jnp.nan),
        center,
        scale,
        jnp.stack(
            decode_mlpe_variances(
                jnp.asarray(raw_variances, dtype=system.unit.dtype), variance_floor=variance_floor
            )
        ),
        _posterior(system, effects[:, 2] - effects[:, :2] @ beta, valid),
    )


def profiled_mlpe_ml_negative_log_likelihood(*args, **kwargs):
    """MLPE negative log likelihood after fitting the signed intercept and slope."""
    return profiled_mlpe_ml_fit(*args, **kwargs)[0]


def mlpe_ml_negative_log_likelihood(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    fixed_effects,
    raw_variances,
    distance_center=None,
    distance_scale=None,
    min_distance_scale=DEFAULT_MIN_DISTANCE_SCALE,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
    pair_mask=None,
):
    """Validation negative log likelihood using fitted coefficients and distance moments."""
    targets, design, system, valid, _, _ = _components(
        distances,
        targets,
        left,
        right,
        n_populations=n_populations,
        raw_variances=raw_variances,
        distance_center=distance_center,
        distance_scale=distance_scale,
        min_distance_scale=min_distance_scale,
        variance_floor=variance_floor,
        jitter=jitter,
        pair_mask=pair_mask,
    )
    residual = targets - design @ jnp.asarray(fixed_effects)
    remaining, effects = system.residual_components(residual[:, None])
    nll = _nll(system, remaining, effects)
    return jnp.where(valid & jnp.isfinite(nll), nll, jnp.nan)


def mlpe_effect_posterior(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    fixed_effects,
    raw_variances,
    distance_center,
    distance_scale,
    min_distance_scale=DEFAULT_MIN_DISTANCE_SCALE,
    variance_floor=DEFAULT_VARIANCE_FLOOR,
    jitter=0.0,
    pair_mask=None,
):
    """Population-effect mean, covariance, and precision factor at current parameters.

    The distance model and variance parameters remain fixed.
    The calculation uses the likelihood's population system and numerical settings.
    """
    targets, design, system, valid, _, _ = _components(
        distances,
        targets,
        left,
        right,
        n_populations=n_populations,
        raw_variances=raw_variances,
        distance_center=distance_center,
        distance_scale=distance_scale,
        min_distance_scale=min_distance_scale,
        variance_floor=variance_floor,
        jitter=jitter,
        pair_mask=pair_mask,
    )
    residual = targets - design @ jnp.asarray(fixed_effects)
    _, effects = system.residual_components(residual[:, None])
    return _posterior(system, effects[:, 0], valid)
