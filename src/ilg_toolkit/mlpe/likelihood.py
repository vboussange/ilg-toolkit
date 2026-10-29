"""Maximum-likelihood population effects (MLPE) likelihood kernels.

Generalized least squares (GLS) profiles the calibration coefficients. Input
array data types set the precision. Invalid inputs, singular systems and
unsupported numerical conditioning return nonfinite results.
"""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from .covariance import _population_covariance_factor


class _MLPEFit(NamedTuple):
    """MLPE coefficients, likelihood and population-effect posterior at supplied variances."""

    nll: jax.Array
    coefficients: jax.Array
    distance_center: jax.Array
    distance_scale: jax.Array
    variances: jax.Array
    posterior: tuple[jax.Array, jax.Array]


def _sample_standardize_distances(distances, *, center=None, scale=None):
    """Standardizes ecological distances and returns them with their center and scale."""
    distances = jnp.asarray(distances)
    count = distances.size
    center = jnp.mean(distances) if center is None else jnp.asarray(center, dtype=distances.dtype)
    differences = distances - center
    scale = (
        jnp.sqrt(jnp.sum(differences**2) / (count - 1))
        if scale is None
        else jnp.asarray(scale, dtype=distances.dtype)
    )
    return differences / scale, center, scale


def _decode_mlpe_variances(raw_variances, *, variance_floor):
    """Decodes raw parameters into the population and residual variances through softplus."""
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
):
    """Returns targets, GLS design, population covariance factor, validity and distance moments."""
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
    standardized, center, scale = _sample_standardize_distances(
        distances, center=distance_center, scale=distance_scale
    )
    design = jnp.stack([jnp.ones_like(standardized), standardized], axis=1)
    unit, residual = _decode_mlpe_variances(raw_variances, variance_floor=variance_floor)
    system = _population_covariance_factor(
        left,
        right,
        n_populations=n_populations,
        unit=unit,
        residual=residual + jitter,
        dtype=distances.dtype,
    )
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
    """Returns the Gaussian negative log likelihood from residual components."""
    quadratic = jnp.sum(remaining**2) / system.residual + jnp.sum(effects**2) / system.unit
    return 0.5 * (system.pair_count * math.log(2 * math.pi) + system.logdet + quadratic)


def _profile(targets, design, system, valid):
    """Profiles the GLS coefficients and returns the likelihood, coefficients and effects."""
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
    """Returns the population-effect posterior mean and covariance."""
    covariance = system.unit * cho_solve(
        (system.factor, True), jnp.eye(len(system.factor), dtype=system.factor.dtype)
    )
    return (
        jnp.where(valid, effects, jnp.nan),
        jnp.where(valid, (covariance + covariance.T) / 2, jnp.nan),
    )


def _profiled_mlpe_ml_fit(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    distance_center=None,
    distance_scale=None,
    min_distance_scale,
    variance_floor,
    jitter,
):
    """Computes the negative log likelihood and signed GLS coefficients of an MLPE fit.

    Inputs with numerical conditioning beyond eps*(1+2*max_degree*u/(e+j)) <= 0.01
    return not-a-number (NaN) results.
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
    )
    nll, beta, _, valid = _profile(targets, design, system, valid)
    return jnp.where(valid, nll, jnp.nan), jnp.where(valid, beta, jnp.nan)


@eqx.filter_jit
def _finalize_mlpe_fit(
    distances,
    targets,
    left,
    right,
    *,
    n_populations,
    raw_variances,
    distance_center=None,
    distance_scale=None,
    min_distance_scale,
    variance_floor,
    jitter,
) -> _MLPEFit:
    """Computes MLPE coefficients, likelihood and population effects at supplied variances.

    The population covariance is factored once for every fitted quantity. Distance
    moments come from the distances, or from the supplied ``distance_center`` and
    ``distance_scale``.
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
    )
    nll, beta, effects, valid = _profile(targets, design, system, valid)
    return _MLPEFit(
        jnp.where(valid, nll, jnp.nan),
        jnp.where(valid, beta, jnp.nan),
        center,
        scale,
        jnp.stack(
            _decode_mlpe_variances(
                jnp.asarray(raw_variances, dtype=system.unit.dtype), variance_floor=variance_floor
            )
        ),
        _posterior(system, effects[:, 2] - effects[:, :2] @ beta, valid),
    )


def _mlpe_ml_negative_log_likelihood(
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
    min_distance_scale,
    variance_floor,
    jitter,
):
    """Computes the negative log likelihood at fixed coefficients and distance moments."""
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
    )
    residual = targets - design @ jnp.asarray(fixed_effects)
    remaining, effects = system.residual_components(residual[:, None])
    nll = _nll(system, remaining, effects)
    return jnp.where(valid & jnp.isfinite(nll), nll, jnp.nan)
